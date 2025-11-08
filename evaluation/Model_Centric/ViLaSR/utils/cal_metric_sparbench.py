from collections import defaultdict
import json

def calculate_average_scores_sparbench(data):
    # 定义分组
    low_types = {
        'depth_prediction_oc', 'depth_prediction_oc_mv',
        'depth_prediction_oo', 'depth_prediction_oo_mv',
        'distance_prediction_oc', 'distance_prediction_oc_mv',
        'distance_prediction_oo', 'distance_prediction_oo_mv'
    }
    
    medium_types = {
        'position_matching', 'camera_motion_infer',
        'view_change_infer'
    }
    
    high_types = {
        'distance_infer_center_oo', 'distance_infer_center_oo_mv',
        'obj_spatial_relation_oc_mv',
        'obj_spatial_relation_oo', 'obj_spatial_relation_oo_mv',
        'spatial_imagination_oc', 'spatial_imagination_oc_mv',
        'spatial_imagination_oo', 'spatial_imagination_oo_mv'
    }
    
    # 统计结构
    group_stats = defaultdict(lambda: {
        'total_score': 0,
        'count': 0,
        'scores': [],
        'types': {}
    })
    
    # 统计数据
    for item in data:
        qtype = item['original_question_type']
        score = item['score']
        
        # 确定分组
        if qtype in low_types:
            group = 'Low'
        elif qtype in medium_types:
            group = 'Medium'
        elif qtype in high_types:
            group = 'High'
        else:
            group = 'Other'

        # if qtype=="depth_prediction_oc":
        #     print(item)
            
        # 更新统计
        group_stats[group]['total_score'] += score
        group_stats[group]['count'] += 1
        group_stats[group]['scores'].append(score)
        if qtype not in group_stats[group]['types']:
            group_stats[group]['types'][qtype] = {'count': 0, 'total_score': 0}
        group_stats[group]['types'][qtype]['count'] += 1
        group_stats[group]['types'][qtype]['total_score'] += score
    
    # 打印结果
    print("\nScore Statistics by Difficulty Groups:")
    print("=" * 60)
    
    total_count = 0
    total_score = 0
    all_type_scores = []  # 存储所有类别的平均分
    for group in ['Low', 'Medium', 'High']:
        if group in group_stats:
            stats = group_stats[group]
            avg_score = stats['total_score'] / stats['count'] if stats['count'] > 0 else 0
            print(f"\n{group} Group:")
            print("-" * 60)
            print(f"Overall: Count = {stats['count']}, Average Score = {avg_score:.3f}")
            print("\nDetailed type statistics:")
            for qtype, type_stat in sorted(stats['types'].items()):
                type_avg = type_stat['total_score'] / type_stat['count']
                all_type_scores.append(type_avg)  # 添加到列表中
                print(f"{qtype:<30} {type_stat['count']:>8} {type_avg:>10.3f}")
            
            total_count += stats['count']
            total_score += stats['total_score']
    
    print("\n" + "=" * 60)
    # 计算所有类别的平均分
    overall_avg = sum(all_type_scores) / len(all_type_scores) if all_type_scores else 0
    print(f"Total: Count = {total_count}, Average Score = {overall_avg:.3f}")

