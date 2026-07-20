import json
import os

def subsample_ytvis_json(input_path, output_path, max_frames=400):
    print(f"⏳ 正在加载原始标注文件: {input_path} ...")
    with open(input_path, 'r') as f:
        data = json.load(f)

    videos = data.get('videos', [])
    annotations = data.get('annotations', [])
    
    print(f"✅ 成功加载！共找到 {len(videos)} 个视频, {len(annotations)} 个实例标注。")
    print(f"⚙️ 开始降采样 (最大帧数限制: {max_frames}) ...\n")

    vid_to_annos = {}
    for anno in annotations:
        vid = anno['video_id']
        if vid not in vid_to_annos:
            vid_to_annos[vid] = []
        vid_to_annos[vid].append(anno)

    processed_video_count = 0
    
    for video in videos:
        original_length = video.get('length', len(video.get('file_names', [])))
        
        if original_length > max_frames:
            keep_indices = [int(round(i * (original_length - 1) / (max_frames - 1))) for i in range(max_frames)]
            
            if 'file_names' in video:
                video['file_names'] = [video['file_names'][i] for i in keep_indices]
            video['length'] = max_frames
            
            if video['id'] in vid_to_annos:
                for anno in vid_to_annos[video['id']]:
                    if 'segmentations' in anno:
                        anno['segmentations'] = [anno['segmentations'][i] for i in keep_indices]
                    if 'bboxes' in anno:
                        anno['bboxes'] = [anno['bboxes'][i] for i in keep_indices]
            
            processed_video_count += 1
            print(f"✂️ 视频 ID={video['id']:<3} | 原长度: {original_length:>4} 帧 -> 降采样至: {max_frames} 帧")

    # ================= 新增：清理全空的幽灵标注 =================
    print("\n🧹 正在清理因降采样导致全空的幽灵标注...")
    valid_annotations = []
    ghost_count = 0
    
    for anno in annotations:
        # 检查这个标注是否还有至少一帧不是 None
        is_valid = False
        if 'segmentations' in anno:
            for seg in anno['segmentations']:
                if seg is not None:
                    is_valid = True
                    break
                    
        if is_valid:
            valid_annotations.append(anno)
        else:
            ghost_count += 1
            
    data['annotations'] = valid_annotations
    print(f"✅ 清理完毕！删除了 {ghost_count} 个全空实例，保留了 {len(valid_annotations)} 个有效实例。")
    # ============================================================

    print(f"\n💾 正在保存至: {output_path} ...")
    with open(output_path, 'w') as f:
        json.dump(data, f)
    print("✅ 保存成功！你可以开始重新评测了。")

if __name__ == "__main__":
    BASE_DIR = "/data/xyjiang/zlni/zlni/data/fishtrack_ytvis/annotations"
    INPUT_JSON = os.path.join(BASE_DIR, "val_long.json")
    OUTPUT_JSON = os.path.join(BASE_DIR, "val.json")
    MAX_FRAMES_LIMIT = 200 
    
    if not os.path.exists(INPUT_JSON):
        print(f"❌ 找不到输入文件: {INPUT_JSON}")
    else:
        subsample_ytvis_json(INPUT_JSON, OUTPUT_JSON, max_frames=MAX_FRAMES_LIMIT)