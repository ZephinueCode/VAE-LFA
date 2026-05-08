import json
from pathlib import Path
import numpy as np

def main():
    runs_to_process = [
        {"name": "Qwen", "report_path": "results/qwen_cycle_vae_off/qwen_cycle_vae_off_report.json"},
        {"name": "Seedream", "report_path": "results/seedream_cycle_vae_on/seedream_cycle_vae_on_report.json"}
    ]
    
    datasets = ["photos", "paintings"]
    categories = ["creature", "architecture", "scenery"]
    
    print("\n" + "="*80)
    print("📊 核心 36 项均值数据导出 (Markdown 格式)")
    print("="*80 + "\n")
    
    print("| 模型 (Model) | 数据集 (Dataset) | 类别 (Category) | 均值 L1 Loss (越小越好) | 均值 SSIM (越大越好) | 均值 LPIPS (越小越好) |")
    print("|---|---|---|---|---|---|")
    
    for run in runs_to_process:
        model_name = run["name"]
        report_file = Path(run["report_path"])
        
        if not report_file.exists():
            print(f"| {model_name} |  报告未找到 | - | - | - | - |")
            continue
            
        with open(report_file, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
            
        for ds in datasets:
            if ds not in report_data:
                continue
                
            for cat in categories:
                if cat not in report_data[ds]:
                    continue
                
                # 用于收集该类别下所有图片、所有 10 轮的数值
                l1_all, ssim_all, lpips_all = [], [], []
                
                images = report_data[ds][cat]
                for img_name, img_data in images.items():
                    if img_data.get("status") == "Failed": 
                        continue
                        
                    manifest_path = Path(img_data.get("pipeline_manifest", ""))
                    if not manifest_path.exists(): continue
                    
                    with open(manifest_path, 'r', encoding='utf-8') as mf:
                        manifest = json.load(mf)
                        
                    summary_path = Path(manifest["metrics"]["summary_path"])
                    if not summary_path.exists(): continue
                        
                    with open(summary_path, 'r', encoding='utf-8') as sf:
                        summary = json.load(sf)
                        
                    for row in summary["rows"]:
                        if "drift_l1_vs_base" in row: l1_all.append(float(row["drift_l1_vs_base"]))
                        if "ssim_vs_base" in row: ssim_all.append(float(row["ssim_vs_base"]))
                        if "lpips_vs_base" in row: lpips_all.append(float(row["lpips_vs_base"]))
                
                # 计算该类的总体均值，保留 4 位小数
                mean_l1 = np.mean(l1_all) if l1_all else 0
                mean_ssim = np.mean(ssim_all) if ssim_all else 0
                mean_lpips = np.mean(lpips_all) if lpips_all else 0
                
                # 打印表格行
                print(f"| {model_name} | {ds} | {cat} | {mean_l1:.4f} | {mean_ssim:.4f} | {mean_lpips:.4f} |")

if __name__ == "__main__":
    main()