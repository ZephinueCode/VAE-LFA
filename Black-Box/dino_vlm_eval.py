import os
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as T
from PIL import Image
try:
    import dashscope
    from dashscope import MultiModalConversation
except ImportError:
    dashscope = None
    MultiModalConversation = None

# ==========================================
# 🛡️ 核心武器 1：指数退避重试机制 (保护 VLM)
# ==========================================
def do_with_retry(func, *args, **kwargs):
    delays = [1, 2, 4, 8, 16, 32, 60]
    for attempt in range(len(delays) + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "too many" in err_str or "rate" in err_str or "throttling" in err_str:
                if attempt < len(delays):
                    sleep_time = delays[attempt]
                    print(f"  ⏳ [API 频控] 触发 429，休眠 {sleep_time} 秒后进行第 {attempt + 1}/{len(delays)} 次重试...")
                    time.sleep(sleep_time)
                    continue
            raise e


def _parse_json_text(content: str) -> dict:
    content = content.replace("```json", "").replace("```", "").strip()
    return json.loads(content)


def vlm_json_call(api_key: str, messages: list[dict], *, tag: str) -> dict:
    if dashscope is None or MultiModalConversation is None:
        raise RuntimeError("dashscope is not installed. Please install dependencies before running VLM evaluation.")
    dashscope.api_key = api_key

    def _call():
        response = MultiModalConversation.call(model='qwen-vl-plus', messages=messages)
        content = response.output.choices[0].message.content[0]['text']
        return _parse_json_text(content)

    try:
        return do_with_retry(_call)
    except Exception as e:
        print(f"  ⚠️ {tag} Failed: {e}")
        raise


def build_judge_specs() -> list[dict[str, Any]]:
    return [
        {
            "key": "instruction_progress",
            "max_score": 40,
            "description": "How much of the intended cumulative edit state is visibly achieved after applying instructions up to the requested round.",
            "subscore_max": 10,
            "subcriteria": [
                ("cumulative_target_match", "How well the final visible state matches the intended cumulative target state up to this round."),
                ("major_edit_completion", "How fully the major requested transformations have been executed."),
                ("attribute_specificity", "How correctly specific requested attributes, materials, colors, styles, or objects are realized."),
                ("spatial_layout_correctness", "How well edited content is placed and integrated in the correct spatial regions and relationships."),
            ],
            "system_prompt": """Role: You are a careful judge for iterative image editing.
Task: Compare the original image and the current edited image. Score only how much of the intended cumulative edit state has been achieved after applying the instructions up to the requested round.

Important reasoning rule:
- The edit sequence is cumulative and later instructions may overwrite earlier ones.
- Do NOT penalize the edited image for failing to preserve an earlier state that should have been replaced by a later instruction.
- Judge the target state after applying the instructions in order up to the requested round.
- Be reasonably forgiving to approximate but clearly visible progress. Do not be overly strict.

Scoring:
- Score each subcriterion independently from 0 to 10.
- Use the full range when needed. Do not default to the same pattern of near-perfect scores unless the visual evidence truly supports it.
- Higher means the visible result better matches the intended cumulative state.

Score anchor guidance for each 0-10 subcriterion:
- 0-2: absent or clearly wrong
- 3-4: weak / barely present
- 5-6: partial but recognizable
- 7-8: strong but imperfect
- 9-10: nearly complete or excellent

Output must be strict JSON:
{"subscores": {"cumulative_target_match": 8, "major_edit_completion": 7, "attribute_specificity": 9, "spatial_layout_correctness": 8}, "reason": "short explanation"}""",
        },
        {
            "key": "state_consistency",
            "max_score": 20,
            "description": "Whether the edited image forms one coherent final state implied by the executed instructions, without obvious contradictions between old and new attributes.",
            "subscore_max": 5,
            "subcriteria": [
                ("contradiction_free_state", "Whether incompatible old and new attributes are avoided."),
                ("global_coherence", "Whether all edited elements belong to one unified overall scene state."),
                ("overwrite_consistency", "Whether later overwrite instructions are resolved cleanly into the current state."),
                ("state_clarity", "Whether the current intended state is visually unambiguous rather than confused or mixed."),
            ],
            "system_prompt": """Role: You are a visual consistency judge for iterative image editing.
Task: Compare the original image and the current edited image. Evaluate whether the edited image looks like one coherent visual state after applying the instructions up to the requested round.

Important reasoning rule:
- The instruction sequence is cumulative and overwrite-aware.
- Later instructions may replace earlier edits. This is expected.
- Penalize contradictions only when the image simultaneously retains incompatible old/new states, or when the current result does not settle into a coherent target state.
- Be reasonably tolerant of imperfect execution if the overall intended state is still clear.

Scoring:
- Score each subcriterion independently from 0 to 5.
- Use the full range when needed. Do not default to the same pattern of high scores unless the evidence strongly supports it.
- Higher means the image expresses a more coherent overwrite-aware target state.

Score anchor guidance for each 0-5 subcriterion:
- 0: failed
- 1: very poor
- 2: weak
- 3: acceptable
- 4: strong
- 5: excellent

Output must be strict JSON:
{"subscores": {"contradiction_free_state": 4, "global_coherence": 4, "overwrite_consistency": 5, "state_clarity": 4}, "reason": "short explanation"}""",
        },
        {
            "key": "visual_quality",
            "max_score": 20,
            "description": "Perceptual image quality, artifact level, and overall visual coherence of the edited image.",
            "subscore_max": 5,
            "subcriteria": [
                ("artifact_control", "Absence of severe artifacts, corruption, or broken regions."),
                ("boundary_cleanliness", "How clean and believable object boundaries and local transitions are."),
                ("texture_material_quality", "How plausible textures and materials appear."),
                ("overall_perceptual_quality", "Overall visual coherence and perceptual appeal."),
            ],
            "system_prompt": """Role: You are a perceptual quality judge for generated images.
Task: Evaluate only the visual quality of the edited image, while using the original image as reference context when helpful.

Judge:
- visual coherence
- artifact severity
- object realism and boundary cleanliness
- texture plausibility
- overall perceptual quality

Important rule:
- Do not over-penalize mild imperfections.
- Reserve very low scores for severe corruption or obvious generation failure.
- The purpose is discrimination, not harsh filtering.

Scoring:
- Score each subcriterion independently from 0 to 5.
- Use the full range when needed and avoid reusing the same near-perfect pattern by default.

Output must be strict JSON:
{"subscores": {"artifact_control": 4, "boundary_cleanliness": 4, "texture_material_quality": 3, "overall_perceptual_quality": 4}, "reason": "short explanation"}""",
        },
        {
            "key": "content_preservation",
            "max_score": 20,
            "description": "How well non-target content, identity, and layout are preserved when they are not supposed to change.",
            "subscore_max": 5,
            "subcriteria": [
                ("identity_preservation", "Whether the main subject identity is preserved when it should remain the same."),
                ("layout_preservation", "Whether unaffected scene structure or layout is preserved."),
                ("collateral_change_control", "Whether unnecessary drift outside the requested edits is minimized."),
                ("edit_locality", "Whether changes are concentrated on intended regions or aspects rather than spilling everywhere."),
            ],
            "system_prompt": """Role: You are a preservation judge for iterative image editing.
Task: Compare the original image and the current edited image. Evaluate whether the image preserves important content that should remain stable while still allowing the requested edits.

Judge:
- preservation of subject identity when identity should remain
- preservation of scene structure/layout when not explicitly replaced
- avoidance of unnecessary collateral changes

Important reasoning rule:
- If the instruction explicitly requests broad scene replacement, do not over-penalize large intended changes.
- Penalize only unnecessary drift beyond the requested edits.
- Be reasonably forgiving when the main editable content is correct and only minor collateral changes occur.

Scoring:
- Score each subcriterion independently from 0 to 5.
- Use the full range when needed. Avoid defaulting to the same pattern of high scores unless preservation is clearly strong.

Output must be strict JSON:
{"subscores": {"identity_preservation": 4, "layout_preservation": 4, "collateral_change_control": 3, "edit_locality": 4}, "reason": "short explanation"}""",
        },
    ]


def build_eval_messages(system_prompt: str, *, img_before_path: str, img_after_path: str, prompts: list[str], mode: str, round_idx: int) -> list[dict]:
    prompts_text = "\n".join([f"Round {i+1}: {p}" for i, p in enumerate(prompts)])
    user_prompt = (
        f"Mode: {mode}\n"
        f"Evaluate the edited image after round {round_idx}.\n"
        f"Instructions executed up to this round:\n{prompts_text}\n"
        "Return strict JSON only."
    )
    return [
        {"role": "user", "content": [
            {"text": system_prompt},
            {"image": f"file://{img_before_path}"},
            {"image": f"file://{img_after_path}"},
            {"text": user_prompt},
        ]}
    ]

def vlm_evaluate(api_key: str, img_before_path: str, img_after_path: str, prompts: list[str], mode: str, round_idx: int) -> dict:
    judge_specs = build_judge_specs()
    breakdown: dict[str, Any] = {}
    total_score = 0

    for spec in judge_specs:
        messages = build_eval_messages(
            spec["system_prompt"],
            img_before_path=img_before_path,
            img_after_path=img_after_path,
            prompts=prompts,
            mode=mode,
            round_idx=round_idx,
        )
        try:
            result = vlm_json_call(api_key, messages, tag=f"Judge `{spec['key']}`")
            raw_subscores = result.get("subscores", {})
            if not isinstance(raw_subscores, dict):
                raise ValueError("Missing or invalid `subscores` field.")
            normalized_subscores: dict[str, int] = {}
            for subkey, _ in spec["subcriteria"]:
                value = int(raw_subscores.get(subkey, 0))
                value = max(0, min(int(spec["subscore_max"]), value))
                normalized_subscores[subkey] = value
            score = sum(normalized_subscores.values())
            reason = str(result.get("reason", "")).strip()
        except Exception as e:
            score = 0
            normalized_subscores = {subkey: 0 for subkey, _ in spec["subcriteria"]}
            reason = f"API call failed: {e}"

        breakdown[spec["key"]] = {
            "score": score,
            "max_score": int(spec["max_score"]),
            "subscores": normalized_subscores,
            "reason": reason,
        }
        total_score += score

    overall_reason = " | ".join(
        f"{key}: {value['reason']}" for key, value in breakdown.items() if value.get("reason")
    )
    return {
        "score": int(total_score),
        "max_score": 100,
        "round_evaluated": int(round_idx),
        "num_instructions_used": len(prompts),
        "breakdown": breakdown,
        "reason": overall_reason,
    }


def vlm_classify_image_type(api_key: str, img_path: str) -> dict:
    system_prompt = """Role: You are an objective image taxonomy annotator for a computer vision benchmark.
Task: Given a single input image, assign it to exactly one of the following two categories.

Category definitions:
1. salient_object:
   The image is primarily scene-dominant or layout-dominant. The visual semantics depend strongly on the broader environment, background, or spatial composition rather than a single foreground object. Typical examples include architecture, landscapes, street scenes, interiors, and other large-scale scenes.

2. clear_object:
   The image contains one clearly identifiable main subject that dominates visual attention. The semantics are mainly determined by this foreground subject rather than the surrounding environment. Typical examples include a single creature, person, product, or other salient object with a relatively simple supporting background.

Instructions:
- Use only the provided image.
- Choose exactly one label: "salient_object" or "clear_object".
- Prefer "clear_object" only when one principal foreground subject is unambiguously dominant.
- Output must be strict JSON.

Required JSON format:
{"label": "salient_object", "reason": "short explanation"}"""

    messages = [
        {"role": "user", "content": [
            {"text": system_prompt},
            {"image": f"file://{img_path}"},
        ]}
    ]

    def _call_vlm():
        result = vlm_json_call(api_key, messages, tag="Image Type Classification")
        label = str(result.get("label", "")).strip()
        if label not in {"salient_object", "clear_object"}:
            raise ValueError(f"Unexpected label: {label}")
        return {
            "label": label,
            "reason": str(result.get("reason", "")).strip(),
        }

    try:
        return do_with_retry(_call_vlm)
    except Exception as e:
        print(f"  ⚠️ Image Type Classification Failed: {e}")
        return {"label": "unknown", "reason": f"API call failed: {e}"}

# ==========================================
# 🛡️ 核心武器 2：DINOv2 图像特征相似度评测
# ==========================================
class DINOEvaluator:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"📦 正在加载 DINOv2 模型到 {self.device}...")
        # 默认使用轻量级 vits14，如果需要更高精度可改为 dinov2_vitb14
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
        self.model.eval()
        self.transform = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    @torch.no_grad()
    def evaluate(self, img1_path: str, img2_path: str) -> float:
        img1 = self.transform(Image.open(img1_path).convert('RGB')).unsqueeze(0).to(self.device)
        img2 = self.transform(Image.open(img2_path).convert('RGB')).unsqueeze(0).to(self.device)
        feat1 = self.model(img1)
        feat2 = self.model(img2)
        sim = torch.nn.functional.cosine_similarity(feat1, feat2).item()
        return round(sim, 4)

# ==========================================
# 🚀 智能匹配 JSON 的神级函数 (完美适配 001/001 套娃路径)
# ==========================================
def find_json_for_run(run_dir: Path, results_root: Path, data_dir: Path) -> Path | None:
    """
    根据相对路径精准扒出 JSON。
    即使路径是 results/.../paintings/scenery/001/001，
    也能准确剥壳，找到 datasets/paintings/scenery/001.json
    """
    try:
        # 获取剥离了 results 根目录后的纯净路径
        # 比如: paintings/scenery/001/001
        rel_path = run_dir.relative_to(results_root)
        
        # 剥掉最后两层重复的文件夹壳 (001/001)，剩下真实的类目层级 (paintings/scenery)
        cat_dir = rel_path.parent.parent 
        
        # 最后一层的名字就是图片名
        sample_name = rel_path.parts[-1] 
        
        # 拼接成最终的 JSON 路径
        json_path = data_dir / cat_dir / f"{sample_name}.json"
        
        if json_path.exists():
            return json_path
            
    except Exception as e:
        print(f"  ⚠️ 路径解析异常: {e}")
        
    return None


def resolve_project_relative(path_str: str | None, results_root: Path) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path if path.exists() else None

    # ReGen-Mech-Interp trace paths are typically relative to the project root
    # that owns `results/`.
    project_root = results_root.parent.parent
    candidate = (project_root / path).resolve()
    if candidate.exists():
        return candidate

    # Fallback: some callers may already provide a path relative to cwd/results root.
    candidate = (results_root / path).resolve()
    if candidate.exists():
        return candidate
    return None


def load_trace_for_run(run_dir: Path) -> dict[str, Any] | None:
    trace_path = run_dir / "metrics_trace.json"
    if not trace_path.exists():
        return None
    try:
        raw = json.loads(trace_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  ⚠️ 读取 metrics_trace.json 失败: {e}")
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def find_final_image(run_dir: Path, trace_data: dict[str, Any] | None, results_root: Path, round_idx: int) -> Path | None:
    if trace_data:
        round_images = trace_data.get("round_images")
        if isinstance(round_images, list) and round_images:
            if 0 <= round_idx < len(round_images):
                resolved = resolve_project_relative(str(round_images[round_idx]), results_root)
                if resolved and resolved.exists():
                    return resolved
            for item in reversed(round_images):
                resolved = resolve_project_relative(str(item), results_root)
                if resolved and resolved.exists():
                    return resolved

    round_tag = f"{round_idx:03d}"
    for name in (f"round_{round_tag}_vae.png", f"round_{round_tag}_raw.png", f"round_{round_tag}.png"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    return None


def collect_prompts(
    *,
    trace_data: dict[str, Any] | None,
    json_file: Path | None,
    mode: str,
) -> tuple[list[str], str | None]:
    if trace_data:
        prompts = trace_data.get("prompts")
        if isinstance(prompts, list) and all(isinstance(item, str) for item in prompts):
            cleaned = [item.strip() for item in prompts]
            if cleaned:
                return cleaned, None

    if json_file is None or not json_file.exists():
        return [], "JSON not found"

    with open(json_file, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)

    if mode == "noop":
        return ["Keep the image unchanged."] * 10, None

    prompts = meta_data.get(mode, meta_data.get("long_chain", []))
    if not isinstance(prompts, list):
        return [], f"Invalid prompt list for mode `{mode}`"
    cleaned = [str(item).strip() for item in prompts if str(item).strip()]
    return cleaned, None


def build_category_summary(eval_report: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for label in ("salient_object", "clear_object"):
        members = [
            item for item in eval_report.values()
            if item.get("image_type", {}).get("label") == label
        ]
        dino_vals = [
            float(item["dino_similarity"])
            for item in members
            if isinstance(item.get("dino_similarity"), (int, float))
        ]
        vlm_vals = []
        per_judge_values: dict[str, list[float]] = {}
        per_judge_subscore_values: dict[str, dict[str, list[float]]] = {}
        for item in members:
            score = item.get("vlm_evaluation", {}).get("score")
            if isinstance(score, (int, float)):
                vlm_vals.append(float(score))
            elif isinstance(score, str) and score.strip().isdigit():
                vlm_vals.append(float(score.strip()))

            breakdown = item.get("vlm_evaluation", {}).get("breakdown", {})
            if isinstance(breakdown, dict):
                for judge_key, judge_payload in breakdown.items():
                    judge_score = judge_payload.get("score") if isinstance(judge_payload, dict) else None
                    if isinstance(judge_score, (int, float)):
                        per_judge_values.setdefault(judge_key, []).append(float(judge_score))
                    if isinstance(judge_payload, dict):
                        subscores = judge_payload.get("subscores", {})
                        if isinstance(subscores, dict):
                            judge_store = per_judge_subscore_values.setdefault(judge_key, {})
                            for subkey, subscore in subscores.items():
                                if isinstance(subscore, (int, float)):
                                    judge_store.setdefault(subkey, []).append(float(subscore))

        summary[label] = {
            "num_samples": len(members),
            "mean_dino_similarity": round(sum(dino_vals) / len(dino_vals), 4) if dino_vals else None,
            "mean_vlm_score": round(sum(vlm_vals) / len(vlm_vals), 4) if vlm_vals else None,
            "mean_vlm_subscores": {
                judge_key: round(sum(values) / len(values), 4) if values else None
                for judge_key, values in per_judge_values.items()
            },
            "mean_vlm_subscore_breakdown": {
                judge_key: {
                    subkey: round(sum(values) / len(values), 4) if values else None
                    for subkey, values in subdict.items()
                }
                for judge_key, subdict in per_judge_subscore_values.items()
            },
        }
    return summary


def create_output_run_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    candidate = base_dir / stem
    suffix = 1
    while candidate.exists():
        candidate = base_dir / f"{stem}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate

def main():
    parser = argparse.ArgumentParser(description="Unified VLM & DINO Evaluator")
    parser.add_argument("--results_dir", type=str, required=True, help="你要评测的根目录 (比如 results/qwen_long_chain_vae_on)")
    parser.add_argument("--data_dir", type=str, default="datasets", help="存放原始 prompt JSON 的数据集目录")
    parser.add_argument("--mode", type=str, choices=["noop", "cycle", "long_chain"], required=True)
    parser.add_argument("--round", type=int, default=10, help="Evaluate the edited image at an arbitrary round number.")
    parser.add_argument("--qwen_api_key", type=str, required=True)
    parser.add_argument("--skip_vlm", action="store_true", help="如果是 noop/cycle，加上这个参数跳过 VLM")
    args = parser.parse_args()

    results_root = Path(args.results_dir)
    data_root = Path(args.data_dir)
    
    if not results_root.exists():
        raise FileNotFoundError(f"找不到结果文件夹: {results_root}")
    if args.round < 1:
        raise ValueError("--round 必须 >= 1")

    output_root = create_output_run_dir(Path("results"))

    report_path = output_root / "eval_report.json"
    summary_path = output_root / "summary.json"
    metadata_path = output_root / "metadata.json"

    eval_report: dict[str, Any] = {}
    metadata = {
        "source_results_dir": str(results_root.resolve()),
        "data_dir": str(data_root.resolve()),
        "mode": args.mode,
        "round_evaluated": int(args.round),
        "skip_vlm": args.skip_vlm,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "vlm_scoring_rubric": {
            "total_score_range": "0-100",
            "judges": build_judge_specs(),
            "notes": [
                "Scoring is round-agnostic and can be applied to any requested round.",
                "Later instructions are allowed to overwrite earlier ones.",
                "The rubric is intentionally not overly strict; approximate but clear progress should receive partial credit.",
            ],
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=4, ensure_ascii=False), encoding="utf-8")

    dino = DINOEvaluator()

    # 🚨 核心魔法：不论目录多深，直接全局抓取所有的 round_000_input 图片
    base_images = list(results_root.rglob("round_000_input.*"))
    print(f"\n🔍 在 {results_root.name} 中共发现 {len(base_images)} 组需要评测的数据。")

    for base_img in base_images:
        run_dir = base_img.parent
        trace_data = load_trace_for_run(run_dir)
        # 取文件夹的名字作为唯一标识符 (比如 "001" 或 "architecture_001")
        if trace_data and isinstance(trace_data.get("sample_id"), str) and trace_data["sample_id"].strip():
            sample_key = trace_data["sample_id"].strip()
        else:
            sample_key = str(run_dir.relative_to(results_root))
        
        if sample_key in eval_report:
            print(f"  ⏭️ {sample_key} 已经评测过，跳过！")
            continue
            
        print(f"\n▶️ 正在评测: {sample_key}")

        # 找终局图像 (优先找 vae 版，如果没有就找 raw 版)
        final_img = find_final_image(run_dir, trace_data, results_root, args.round)
        if final_img is None or not final_img.exists():
            print(f"  ⚠️ 找不到 {sample_key} 的第 {args.round} 轮输出图，跳过。")
            continue

        # 1. 跑 DINO 评测 (所有模式都跑)
        print(f"  🦕 计算 DINOv2 相似度...")
        dino_score = dino.evaluate(str(base_img), str(final_img))
        print(f"    -> DINO Cosine Sim: {dino_score}")

        print(f"  🧭 请求 Qwen-VL 进行图像类型分类...")
        image_type = vlm_classify_image_type(args.qwen_api_key, str(base_img))
        print(f"    -> Image Type: {image_type.get('label', 'unknown')}")

        vlm_res = {"score": "N/A", "reason": "Skipped (flag --skip_vlm active)"}
        
        # 2. 跑 VLM 评测 (如果没跳过的话)
        if not args.skip_vlm:
            trace_meta = None
            if trace_data and isinstance(trace_data.get("meta_path"), str):
                trace_meta = resolve_project_relative(trace_data["meta_path"], results_root)
            json_file = trace_meta or find_json_for_run(run_dir, results_root, data_root)
            prompts, prompt_err = collect_prompts(trace_data=trace_data, json_file=json_file, mode=args.mode)

            if prompt_err is not None:
                print(f"  ⚠️ 找不到 {sample_key} 对应的 Prompt 信息，跳过 VLM。")
                vlm_res = {"score": 0, "reason": prompt_err}
            elif len(prompts) < args.round:
                print(f"  ⚠️ Prompt 数不足 {args.round} 条，跳过 VLM。")
                vlm_res = {"score": 0, "reason": "Insufficient prompts"}
            else:
                print(f"  🤖 请求 Qwen-VL 打分...")
                vlm_res = vlm_evaluate(
                    args.qwen_api_key,
                    str(base_img),
                    str(final_img),
                    prompts[: args.round],
                    args.mode,
                    args.round,
                )
                print(f"    -> VLM Score: {vlm_res.get('score', 0)}")

        # 3. 记录并立即存档
        eval_report[sample_key] = {
            "dino_similarity": dino_score,
            "image_type": image_type,
            "vlm_evaluation": vlm_res,
            "paths": {
                "base_image": str(base_img),
                "final_image": str(final_img)
            },
            "trace_source": "metrics_trace.json" if trace_data else "legacy_path_inference",
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(eval_report, f, indent=4, ensure_ascii=False)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    "metadata": metadata,
                    "category_summary": build_category_summary(eval_report),
                    "num_samples_finished": len(eval_report),
                },
                f,
                indent=4,
                ensure_ascii=False,
            )

    final_summary = {
        "metadata": metadata,
        "category_summary": build_category_summary(eval_report),
        "num_samples_finished": len(eval_report),
        "report_path": str(report_path),
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, indent=4, ensure_ascii=False)

    print(f"\n🎉 评测全部完成！")
    print(f"📄 明细报告: {report_path}")
    print(f"📊 分类汇总: {summary_path}")
    print(f"📁 输出目录: {output_root}")

if __name__ == "__main__":
    main()
