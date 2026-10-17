# track_state.py
# Copyright (c) 2026. All rights reserved.
# Written for DVIS_FBFL High-Precision Video Object Tracking & Segmentation.

from typing import Tuple, List, Optional
import torch


class TrackStateTensor:
    """
    工业级、高精度、完全可微分的 8D 卡尔曼滤波（Kalman Filter）时序轨迹状态跟踪器。
    
    采用视觉跟踪 (如 ByteTrack, DeepSORT) 标准的 8D 状态空间表示：
        x = [xc, yc, a, h, v_xc, v_yc, v_a, v_h]^T
    其中 (xc, yc) 为中心点坐标，a 为宽高比 (w/h)，h 为高度，后四项为对应的速度分量。
    
    核心学术优势：
    1. 【数学严谨性】：内置完整的 8x8 状态协方差矩阵 P，过程噪声协方差 Q，以及 4x8 的观测投影矩阵 H。
    2. 【自适应卡尔曼增益控制】：根据当前帧 Mask 分割的面积和分类置信度，动态调整测量噪声协方差矩阵 R：
       当目标发生遮挡、漏检（置信度变小或面积接近0）时，测量的协方差 R 趋于无限大，卡尔曼增益 K 自动逼近于 0，
       更新过程完全转由运动模型物理外推（预测值）主导。
    3. 【完全可微】：所有矩阵逆（Inverse）和相乘均通过 PyTorch Native 算子实现，梯度流完全打通。
    4. 【物理阻尼约束（Friction）】：当物体处于连续丢失状态（lost_streak > 0）时，其速度分量会自动乘以阻尼系数，
       防止物体在完全遮挡时无休止、线性地暴走出画面边缘。
    """
    def __init__(
        self, 
        track_id: int, 
        query_embd: torch.Tensor, 
        mask: torch.Tensor, 
        bbox: torch.Tensor, 
        class_id: int, 
        matched_gt_id: int = -1,
        std_weight_position: float = 0.05,     # 观测位置的标准差权重系数 (基于高度 h 缩放)
        std_weight_velocity: float = 0.00625,   # 预测速度的标准差权重系数 (基于高度 h 缩放)
        friction_decay: float = 0.95            # 丢失状态下的速度阻尼衰减系数
    ):
        self.track_id = track_id
        self.query_embd = query_embd             # 语义 Query 特征 [C]
        self.last_mask = mask                   # 上一帧的最优 Mask [H, W]
        self.class_id = class_id
        self.matched_gt_id = matched_gt_id
        self.friction = friction_decay
        self.device = bbox.device
        
        self.age = 1
        self.lost_streak = 0
        
        # 保存基础的标准差配置
        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity

        # 1. 物理转换矩阵初始化
        self._init_kalman_matrices()

        # 2. 测量转换：将输入 Bbox (x1, y1, x2, y2) 转化为卡尔曼空间测度 z = [xc, yc, a, h]^T
        z = self._bbox_to_measurement(bbox)
        
        # 3. 状态初始化：[xc, yc, a, h, 0, 0, 0, 0]^T, 初始速度分量为 0
        self.x = torch.zeros(8, device=self.device)
        self.x[:4] = z

        # 4. 初始化状态协方差矩阵 P (8x8)
        self.P = torch.eye(8, device=self.device)
        # 位置噪声基于初始高度初始化，速度噪声初始化较大
        h = z[3]
        pos_var = (self._std_weight_position * h) ** 2
        vel_var = (self._std_weight_velocity * h) ** 2
        self.P[0:4, 0:4] = torch.eye(4, device=self.device) * pos_var
        self.P[4:8, 4:8] = torch.eye(4, device=self.device) * vel_var

        # 暂存当前帧的最优框 (x1, y1, x2, y2)
        self.last_bbox = bbox

    def _init_kalman_matrices(self):
        """
        初始化 8D 卡尔曼滤波的标准矩阵。
        """
        # 状态转移矩阵 F (8x8)，设 dt = 1
        self.F = torch.eye(8, device=self.device)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # 观测投影矩阵 H (4x8)：将 8D 状态空间映射到 4D 测量坐标系
        self.H = torch.zeros(4, 8, device=self.device)
        for i in range(4):
            self.H[i, i] = 1.0

    def _bbox_to_measurement(self, bbox: torch.Tensor) -> torch.Tensor:
        """
        将 [x1, y1, x2, y2] 坐标转换为归一化卡尔曼测度 [xc, yc, a, h]^T
        """
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        w = x2 - x1
        h = y2 - y1
        xc = (x1 + x2) / 2.0
        yc = (y1 + y2) / 2.0
        a = w / (h + 1e-6) # 宽高比 w / h
        return torch.stack([xc, yc, a, h])

    def _measurement_to_bbox(self, state_x: torch.Tensor) -> torch.Tensor:
        """
        [可微转换]：将 8D 状态向量的前四项 [xc, yc, a, h] 还原为物理 Bbox [x1, y1, x2, y2]
        """
        xc, yc, a, h = state_x[0], state_x[1], state_x[2], state_x[3]
        w = a * h
        x1 = xc - w / 2.0
        y1 = yc - h / 2.0
        x2 = xc + w / 2.0
        y2 = yc + h / 2.0
        return torch.clamp(torch.stack([x1, y1, x2, y2]), min=0.0, max=1.0)

    def _get_process_noise(self, h: torch.Tensor) -> torch.Tensor:
        """
        计算状态过程噪声协方差 Q (8x8)。
        位置和速度的变化噪声幅值会随目标高度 h 的变化自适应缩放（具备空间尺度不变性）。
        """
        Q = torch.eye(8, device=self.device)
        pos_std = self._std_weight_position * h
        vel_std = self._std_weight_velocity * h
        
        Q[0:4, 0:4] = torch.eye(4, device=self.device) * (pos_std ** 2)
        Q[4:8, 4:8] = torch.eye(4, device=self.device) * (vel_std ** 2)
        return Q

    def _get_measurement_noise(self, h: torch.Tensor, confidence: float, area_weight: torch.Tensor) -> torch.Tensor:
        """
        [自适应卡尔曼设计核心]：计算动态测量噪声协方差 R (4x4)。
        
        根据置信度 (confidence) 与 Mask 有效面积 (area_weight) 动态调节测量置信标准差。
        如果当前分割由于遮挡漏检导致非常嘈杂（confidence/area 小），R 对应的值会几何倍率上升，
        使系统选择相信运动预测而不是损坏的测量。
        """
        R = torch.eye(4, device=self.device)
        # 基础测量标准差，依据高度 h 自适应缩放
        std_pos = self._std_weight_position * h
        
        # 自适应尺度门控因子（确保分母不为 0）
        gate = torch.clamp(confidence * area_weight, min=1e-5)
        # 噪声标准差与可信度成反比
        adapted_std_pos = std_pos / gate
        
        R[0:2, 0:2] = torch.eye(2, device=self.device) * (adapted_std_pos ** 2)
        # 宽高比 a 的扰动通常较位置更敏感，稍微放大其噪声方差
        R[2, 2] = (adapted_std_pos * 1.5) ** 2
        R[3, 3] = (adapted_std_pos) ** 2
        return R

    def predict_next_bbox(self) -> torch.Tensor:
        """
        [时序物理外推预测]
        执行卡尔曼滤波的前向预测步（Prediction Step），并投影输出预测的 4D 物理 Bbox。
        """
        # 1. 运动外推更新：x_{t|t-1} = F * x_{t-1|t-1}
        pred_x = torch.mv(self.F, self.x)

        # 如果当前目标处于丢失状态，对其速度项应用阻尼衰减，防止预测无限向边缘飘走
        if self.lost_streak > 0:
            pred_x[4:8] = pred_x[4:8] * (self.friction ** self.lost_streak)

        # 2. 预测状态协方差更新：P_{t|t-1} = F * P_{t-1|t-1} * F^T + Q
        h = pred_x[3]  # 当前预测的高度项
        Q = self._get_process_noise(h)
        pred_P = torch.mm(torch.mm(self.F, self.P), self.F.T) + Q

        # 3. 将卡尔曼估计出来的 [xc, yc, a, h] 投影成 [x1, y1, x2, y2]
        predicted_bbox = self._measurement_to_bbox(pred_x)
        return predicted_bbox

    def update(
        self, 
        query_embd: torch.Tensor, 
        mask: torch.Tensor, 
        bbox: torch.Tensor, 
        matched_gt_id: int = -1,
        mask_confidence: float = 1.0
    ):
        """
        [端到端可微分更新] 
        执行卡尔曼滤波的更新校准步（Update Step），动态调配测量增益。
        
        参数:
          query_embd: 当前帧观测到的 Query 语义特征向量 [C]
          mask: 当前帧分割生成的概率/二值 Mask [H, W]
          bbox: 当前帧分割出的检测边界框坐标 [4] -> (x1, y1, x2, y2)
          mask_confidence: 类别/实例置信度 [0.0, 1.0]
        """
        self.query_embd = query_embd
        self.last_mask = mask
        self.matched_gt_id = matched_gt_id

        # -----------------------------------------------------------------
        # Step 1: 物理预测步预演 (获得时序预测状态)
        # -----------------------------------------------------------------
        pred_x = torch.mv(self.F, self.x)
        h_pred = pred_x[3]
        Q = self._get_process_noise(h_pred)
        pred_P = torch.mm(torch.mm(self.F, self.P), self.F.T) + Q

        # -----------------------------------------------------------------
        # Step 2: 动态评估观测质量 (Gated Measurement Quality)
        # -----------------------------------------------------------------
        mask_area = mask.sum()
        H_img, W_img = mask.shape
        # 计算当前测量 Mask 的物理面积比例，用于衡量分割热力是否存在塌陷
        area_weight = torch.clamp(mask_area / (H_img * W_img * 0.0001 + 1e-6), max=1.0)
        
        # 转化测量值
        z = self._bbox_to_measurement(bbox)

        # -----------------------------------------------------------------
        # Step 3: 标准卡尔曼矩阵计算更新 (完全可微)
        # -----------------------------------------------------------------
        # 3.1 动态噪声调整
        R = self._get_measurement_noise(h_pred, mask_confidence, area_weight)
        
        # 3.2 测量偏差（残差）：y = z - H * x
        y = z - torch.mv(self.H, pred_x)
        
        # 3.3 残差协方差：S = H * P * H^T + R
        S = torch.mm(torch.mm(self.H, pred_P), self.H.T) + R
        
        # 3.4 求解卡尔曼增益：K = P * H^T * S^-1 (PyTorch 可微求逆)
        K = torch.mm(torch.mm(pred_P, self.H.T), torch.inverse(S))
        
        # 3.5 状态后验更新：x_t = x_prediction + K * y
        self.x = pred_x + torch.mv(K, y)
        
        # 3.6 协方差后验更新：P_t = (I - K * H) * P_prediction
        I = torch.eye(8, device=self.device)
        self.P = torch.mm(I - torch.mm(K, self.H), pred_P)

        # -----------------------------------------------------------------
        # Step 4: 投影并同步更新边界框和状态年龄
        # -----------------------------------------------------------------
        self.last_bbox = self._measurement_to_bbox(self.x)
        self.history_bboxes.append(self.last_bbox)
        if len(self.history_bboxes) > 5:
            self.history_bboxes.pop(0)

        # 更新丢失状态计时器
        if mask_area < 1.0:
            self.lost_streak += 1
        else:
            self.lost_streak = 0
            
        self.age += 1