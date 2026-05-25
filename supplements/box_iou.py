

def box_iou(bbox1, bbox2):
    x1min, y1min, x1max, y1max = bbox1
    x2min, y2min, x2max, y2max = bbox2
    
    size_1 = (x1max - x1min) * (y1max - y1min)
    size_2 = (x2max - x2min) * (y2max - y2min)
    
    inter_width = max(min(x1max, x2max) - max(x1min, x2min), 0)
    inter_height = max(min(y1max, y2max) - max(y1min, y2min), 0)
    inter_size = inter_width * inter_height

    union_size = size_1 + size_2 - inter_size

    return inter_size / union_size
