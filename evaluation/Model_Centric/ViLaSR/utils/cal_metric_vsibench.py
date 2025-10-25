import json
from collections import defaultdict

# 定义类别顺序字典，使用简短名称
QUESTION_TYPES = {
    'object_rel_distance': 'Rel. Dist',
    'object_rel_direction': 'Rel. Dir',
    'route_planning': 'Route Plan',
    'obj_appearance_order': 'Appr. Order',
    # 分隔
    'object_counting': 'Obj. Count',
    'object_abs_distance': 'Abs. Dist',
    'object_size_estimation': 'Obj. Size',
    'room_size_estimation': 'Room Size'
}

def merge_difficulty_levels(type_scores):
    merged_scores = defaultdict(list)
    for question_type, scores in type_scores.items():
        if any(diff in question_type for diff in ['_easy', '_medium', '_hard']):
            base_type = question_type.split('_easy')[0].split('_medium')[0].split('_hard')[0]
            merged_scores[base_type].extend(scores)
        else:
            merged_scores[question_type].extend(scores)
    return merged_scores

def calculate_average_scores_vsibench(data):
    type_scores = defaultdict(list)
    all_scores = []
    
    for item in data:
        type_scores[item['original_question_type']].append(item['score'])
        all_scores.append(item['score'])
    
    merged_scores = merge_difficulty_levels(type_scores)
    
    # 按指定顺序输出
    print("\nType           Score   Count")
    print("-" * 35)
    type_averages = {}
    
    # 第一组
    first_group = ['object_rel_distance', 'object_rel_direction', 
                   'route_planning', 'obj_appearance_order']
    for qtype in first_group:
        if qtype in merged_scores:
            scores = merged_scores[qtype]
            avg_score = sum(scores) / len(scores)
            type_averages[qtype] = avg_score
            print(f"{QUESTION_TYPES[qtype]:<14} {avg_score:.3f}   {len(scores)}")
    
    # 分隔线
    print("-" * 35)
    
    # 第二组
    second_group = ['object_counting', 'object_abs_distance',
                    'object_size_estimation', 'room_size_estimation']
    for qtype in second_group:
        if qtype in merged_scores:
            scores = merged_scores[qtype]
            avg_score = sum(scores) / len(scores)
            type_averages[qtype] = avg_score
            print(f"{QUESTION_TYPES[qtype]:<14} {avg_score:.3f}   {len(scores)}")
    
    overall_score = sum(type_averages.values()) / len(type_averages)
    print("-" * 35)
    print(f"Overall        {overall_score:.3f}   {len(all_scores)}")

