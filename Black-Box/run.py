import os
import json
import argparse
from pathlib import Path
import dashscope
from dashscope import MultiModalConversation

from src.inference.pipeline import run_inference 

def vlm_evaluate(api_key: str, img_before_path: str, img_after_path: str, prompts: list[str], mode: str) -> dict:
    dashscope.api_key = api_key
    prompts_text = "\n".join([f"Round {i+1}: {p}" for i, p in enumerate(prompts)])
    system_prompt = """Role: You are an objective computer vision evaluator.
Task: Compare Image 1 (original) and Image 2 (final after 10 edits). The user provided a sequence of 10 editing instructions.
Scoring Criteria:
1. Base Score (0-10): Count EXACTLY how many of the 10 specific instructions are visibly and successfully executed in Image 2. (e.g., if 6 instructions are clearly visible, your base score is 6).
2. Quality Adjustment: Adjust the base score by -2 to +2 based on image quality. Deduct points if the image has severe artifacts, blurring, or destroyed unrelated background elements. Add points if the image remains exceptionally coherent and high-quality.
3. Final Score: Must be an integer between 1 and 10.
Output Format: MUST strictly be JSON. Example: {"score": 7, "reason": "Successfully completed 7 out of 10 steps. Minor background artifacts caused a 1-point deduction."}"""
    user_prompt = f"Mode: {mode}\nSequence of instructions executed:\n{prompts_text}\nPlease provide your score in strict JSON format:"
    messages = [
        {"role": "user", "content": [
            {"text": system_prompt},
            {"image": f"file://{img_before_path}"},
            {"image": f"file://{img_after_path}"},
            {"text": user_prompt}
        ]}
    ]
    try:
        response = MultiModalConversation.call(model='qwen-vl-plus', messages=messages)
        content = response.output.choices[0].message.content[0]['text']
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        print(f"VLM Evaluation Failed: {e}")
        return {"score": 0, "reason": "API call failed or parsing error"}

class ConfigDict(dict):
    def __init__(self, d=None):
        super().__init__()
        if d is None: d = {}
        for k, v in d.items():
            if isinstance(v, dict): self[k] = ConfigDict(v)
            elif isinstance(v, list): self[k] = [ConfigDict(i) if isinstance(i, dict) else i for i in v]
            else: self[k] = v
    def __getattr__(self, key):
        if key in self: return self[key]
        raise AttributeError(f"Configuration missing key: {key}")
    def __setattr__(self, key, value):
        self[key] = value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["qwen", "seedream"], required=True)
    parser.add_argument("--mode", type=str, choices=["noop", "cycle", "longchain"], required=True)
    parser.add_argument("--use_vae", action="store_true")
    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--qwen_api_key", type=str, required=True)
    args = parser.parse_args()

    config_path = "src/config/config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config_dict = json.load(f)

    datasets = ["photos", "paintings"]
    categories = ["creature", "architecture", "scenery"]
    
    vae_status = "vae_on" if args.use_vae else "vae_off"
    global_run_name = f"{args.model}_{args.mode}_{vae_status}"
    results_root = Path("results") / global_run_name
    report_path = results_root / f"{global_run_name}_report.json"
    
    # 断点续传
    if report_path.exists():
        print(f"\n检测到历史报告，启动断点续传: {report_path.name}")
        with open(report_path, 'r', encoding='utf-8') as f:
            final_summary_report = json.load(f)
    else:
        final_summary_report = {}

    for ds in datasets:
        if ds not in final_summary_report:
            final_summary_report[ds] = {}
        for cat in categories:
            if cat not in final_summary_report[ds]:
                final_summary_report[ds][cat] = {}
            print(f"\nStart Processing: [{ds}] - [{cat}]")
            
            cat_dir = Path(args.data_dir) / ds / cat
            if not cat_dir.exists(): continue
            
            images = sorted([f for f in cat_dir.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])[:10]
            
            for img_path in images:
                sample_name = img_path.stem
                
                # 如果在报告里查到这张图直接跳过
                if sample_name in final_summary_report[ds][cat]:
                    print(f"{sample_name} 已经处理过，直接跳过！")
                    continue
                
                json_path = cat_dir / f"{sample_name}.json"
                if not json_path.exists(): continue
                with open(json_path, 'r', encoding='utf-8') as f:
                    meta_data = json.load(f)
                    
                prompts = ["Keep the image unchanged. Do not edit anything."] * 10 if args.mode == "noop" else meta_data.get(args.mode, [])
                if len(prompts) < 10: continue

                print(f"  -> Generating sequence for {sample_name}...")
                
                final_config = ConfigDict(raw_config_dict)
                final_config.api.api_key = final_config.api.get("api_key_env", "")
                
                final_config.run.input_dir = None
                final_config.run.input_image = img_path
                final_config.run.results_dir = results_root / ds / cat
                final_config.run.run_name = sample_name
                final_config.run.rounds = 10
                final_config.prompts.per_round = prompts
                final_config.vae.enabled = args.use_vae
                final_config.intervention.enabled = args.use_vae

                if args.model == "qwen":
                    final_config.api.base_url = "https://dashscope.aliyuncs.com/api/v1"
                    final_config.api.api_key_env = args.qwen_api_key
                    final_config.api.api_key = args.qwen_api_key 
                    final_config.api.model = "qwen-image-2.0"
                    final_config.api.dashscope_endpoint = "/services/aigc/multimodal-generation/generation"
                    final_config.api.dashscope_parameters = {"n": 1}

                # 失败跳过并存档
                try:
                    run_result = run_inference(final_config)
                except Exception as e:
                    print(f"  ❌ {sample_name} 运行失败 (可能是安全风控拦截): {e}")
                    final_summary_report[ds][cat][sample_name] = {"status": "Failed", "error_reason": str(e)}
                    with open(report_path, 'w', encoding='utf-8') as f:
                        json.dump(final_summary_report, f, indent=4, ensure_ascii=False)
                    continue 
                
                img_run_dir = Path(run_result["results"][0]["run_dir"])
                base_img = img_run_dir / f"round_000_input{img_path.suffix.lower()}"
                final_img = img_run_dir / "round_010_raw.png"
                if args.use_vae:
                    final_img = img_run_dir / "round_010_vae.png"

                manifest_path = img_run_dir / "manifest.json"
                if args.mode == "longchain":
                    print(f"Requesting Qwen-VL for VLM Score...")
                    vlm_res = vlm_evaluate(args.qwen_api_key, str(base_img), str(final_img), prompts, args.mode)
                    print(f"{sample_name} finished! VLM Score: {vlm_res.get('score', 0)}")
                else:
                    print(f"模式为 {args.mode}，跳过 VLM 评测，留存给后续 DINO 处理。")
                    vlm_res = {"score": "N/A", "reason": f"Skipped for {args.mode} mode. Awaiting DINO."}
                
                final_summary_report[ds][cat][sample_name] = {
                    "vlm_evaluation": vlm_res,
                    "pipeline_manifest": str(manifest_path)
                }
                
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(final_summary_report, f, indent=4, ensure_ascii=False)

    print(f"\n完美收工！报告路径: {report_path}")

if __name__ == "__main__":
    main()