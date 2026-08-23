import datetime

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from camera import HeightCamera


def trimmed_mean(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    low, high = np.percentile(values, [5, 95])
    selected = values[(values >= low) & (values <= high)]
    if selected.size == 0:
        selected = values
    return float(np.mean(selected))


def pixel_to_point(u, v, depth_mm, intrinsics):
    z_m = depth_mm / 1000.0
    x_m = (float(u) - intrinsics.ppx) * z_m / intrinsics.fx
    y_m = (float(v) - intrinsics.ppy) * z_m / intrinsics.fy
    return [x_m * 1000.0, y_m * 1000.0, z_m * 1000.0]


def fit_plane_robust(points):
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 12:
        raise ValueError("平面点不足，无法拟合参考平面")

    center = np.median(points, axis=0)
    for _ in range(3):
        centered = points - center
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        normal = vt[-1]
        distances = np.abs(centered @ normal)
        if distances.size < 12:
            break
        median_distance = float(np.median(distances))
        threshold = max(2.0, 3.0 * median_distance)
        inliers = points[distances < threshold]
        if inliers.shape[0] < 12:
            break
        points = inliers
        center = np.mean(points, axis=0)

    centered = points - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    if normal[2] < 0:
        normal = -normal
    d = float(-normal @ center)
    return normal, d


def distances_to_plane(points, normal, d):
    return np.asarray(points, dtype=np.float64) @ normal + d


def measure_height(depth_mm, intrinsics, surround_margin_px=80):
    depth = np.asarray(depth_mm, dtype=np.float32)
    valid_mask = (depth > 100.0) & (depth < 1800.0) & np.isfinite(depth)
    if np.count_nonzero(valid_mask) < 100:
        raise ValueError("有效深度点太少，请确保物体和平面都在画面中")

    valid_values = depth[valid_mask]
    low_all, high_all = np.percentile(valid_values, [1, 99])
    clean_mask = valid_mask & (depth >= low_all) & (depth <= high_all)
    clean_values = depth[clean_mask]
    if clean_values.size < 100:
        raise ValueError("剔除异常点后有效数据不足")

    clean_u16 = np.clip(clean_values, 0, 65535).astype(np.uint16)
    threshold_value, _ = cv2.threshold(
        clean_u16.reshape(-1, 1),
        0,
        1,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    object_mask = clean_mask & (depth <= float(threshold_value))
    plane_mask = clean_mask & (depth > float(threshold_value))

    if np.count_nonzero(object_mask) < 50 or np.count_nonzero(plane_mask) < 50:
        raise ValueError("无法自动区分物体和平面，请调整物体位置或光线")

    kernel = np.ones(
        (surround_margin_px * 2 + 1, surround_margin_px * 2 + 1),
        np.uint8,
    )
    object_dilated = cv2.dilate(
        object_mask.astype(np.uint8), kernel, iterations=1
    )
    local_plane_mask = plane_mask & (object_dilated > 0)
    if np.count_nonzero(local_plane_mask) < 50:
        raise ValueError("物体周围平面点不足，请调整物体位置")

    plane_uv = np.argwhere(local_plane_mask)
    if plane_uv.shape[0] < 12:
        raise ValueError("局部平面点不足")
    plane_points = np.array(
        [
            pixel_to_point(int(uv[1]), int(uv[0]), float(depth[uv[0], uv[1]]), intrinsics)
            for uv in plane_uv
        ],
        dtype=np.float64,
    )
    normal, d = fit_plane_robust(plane_points)

    clean_y, clean_x = np.where(clean_mask)
    clean_depth = depth[clean_mask]
    clean_z_mm = clean_depth.astype(np.float64)
    clean_x_mm = (clean_x.astype(np.float64) - intrinsics.ppx) * clean_z_mm / intrinsics.fx
    clean_y_mm = (clean_y.astype(np.float64) - intrinsics.ppy) * clean_z_mm / intrinsics.fy
    clean_points = np.column_stack([clean_x_mm, clean_y_mm, clean_z_mm])
    plane_distances = clean_points @ normal + d

    height_map = np.full(depth.shape, np.nan, dtype=np.float64)
    height_map[clean_mask] = plane_distances

    candidate_mask = (
        (np.abs(height_map) > 2.0)
        & (np.abs(height_map) < 600.0)
        & (object_dilated > 0)
    )
    if np.count_nonzero(candidate_mask) < 12:
        raise ValueError("物体上表面点不足，请调整物体或平面")

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate_mask.astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        raise ValueError("未找到连续物体区域")
    largest_label = int(
        np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    )
    object_final_mask = (labels == largest_label)
    if np.count_nonzero(object_final_mask) < 12:
        raise ValueError("物体区域过小")

    object_distances = np.abs(height_map[object_final_mask])
    object_distances = object_distances[np.isfinite(object_distances)]
    if object_distances.size < 12:
        raise ValueError("物体高度点不足")

    low, high = np.percentile(object_distances, [5, 95])
    selected = object_distances[
        (object_distances >= low) & (object_distances <= high)
    ]
    if selected.size < 6:
        selected = object_distances

    # 取距离参考平面最高的 20% 点，作为物体上表面。
    top_cut = float(np.percentile(selected, 80))
    top_points = selected[selected >= top_cut]
    if top_points.size < 3:
        top_points = selected
    height_mm = float(np.mean(top_points))

    object_depth = trimmed_mean(depth[object_mask])
    plane_depth = trimmed_mean(depth[local_plane_mask])
    return {
        "object_depth_mm": object_depth,
        "plane_depth_mm": plane_depth,
        "height_mm": height_mm,
        "object_points": int(np.count_nonzero(object_final_mask)),
        "plane_points": int(np.count_nonzero(local_plane_mask)),
    }


class HeightMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("D435 物体高度测量")
        self.resize(1320, 820)
        self.camera = None
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        control = QGroupBox("控制")
        control_layout = QVBoxLayout(control)
        self.camera_btn = QPushButton("启动相机")
        self.measure_btn = QPushButton("测量物体高度")
        control_layout.addWidget(self.camera_btn)
        control_layout.addWidget(self.measure_btn)
        control_layout.addStretch(1)
        root.addWidget(control)

        view_panel = QWidget()
        view_layout = QVBoxLayout(view_panel)
        view_layout.setContentsMargins(0, 0, 0, 0)

        image_row = QHBoxLayout()
        color_box = QVBoxLayout()
        color_box.addWidget(QLabel("RGB 画面"))
        self.color_view = QLabel("等待相机...")
        self.color_view.setAlignment(Qt.AlignCenter)
        self.color_view.setMinimumSize(520, 340)
        self.color_view.setStyleSheet(
            "background:#111827; color:#D1D5DB; border-radius:6px;"
        )
        color_box.addWidget(self.color_view, 1)

        depth_box = QVBoxLayout()
        depth_box.addWidget(QLabel("深度点云视图"))
        self.depth_view = QLabel("等待相机...")
        self.depth_view.setAlignment(Qt.AlignCenter)
        self.depth_view.setMinimumSize(520, 340)
        self.depth_view.setStyleSheet(
            "background:#111827; color:#D1D5DB; border-radius:6px;"
        )
        depth_box.addWidget(self.depth_view, 1)

        image_row.addLayout(color_box, 1)
        image_row.addLayout(depth_box, 1)
        view_layout.addLayout(image_row, 2)

        self.result_label = QLabel("高度：未测量")
        self.result_label.setStyleSheet(
            "font-size:18px; font-weight:bold; color:#0066CC; padding:8px;"
        )
        view_layout.addWidget(self.result_label)

        log_group = QGroupBox("日志")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(
            "QPlainTextEdit { background:#1E1E1E; color:#D1D5DB;"
            " font-family:Consolas; }"
        )
        log_layout.addWidget(self.log_text)
        view_layout.addWidget(log_group, 1)

        root.addWidget(view_panel, 1)

        self.camera_btn.clicked.connect(self._toggle_camera)
        self.measure_btn.clicked.connect(self._measure)

    def _toggle_camera(self):
        if self.camera is not None and self.camera.isRunning():
            self.camera.stop()
            self.camera = None
            self.camera_btn.setText("启动相机")
            return

        self.camera = HeightCamera()
        self.camera.color_ready.connect(self._update_color)
        self.camera.depth_ready.connect(self._update_depth)
        self.camera.status_changed.connect(
            lambda text: self._log("相机状态：" + text)
        )
        self.camera.error_signal.connect(self._log)
        self.camera.start()
        self.camera_btn.setText("关闭相机")

    def _measure(self):
        if self.camera is None or not self.camera.isRunning():
            self._log("请先启动相机")
            return
        depth = self.camera.latest_depth_mm()
        if depth is None:
            self._log("尚未获取到深度数据")
            return
        intrinsics = self.camera.get_color_intrinsics()
        if intrinsics is None:
            self._log("相机内参不可用")
            return
        try:
            result = measure_height(depth, intrinsics)
        except Exception as exc:
            self._log("测量失败: %s" % exc)
            self.result_label.setText("高度：测量失败")
            return

        self.result_label.setText(
            "高度：%.2f mm | 物体深度 %.2f mm | 平面深度 %.2f mm"
            % (
                result["height_mm"],
                result["object_depth_mm"],
                result["plane_depth_mm"],
            )
        )
        self._log(
            "物体点 %d，平面点 %d，物体深度 %.2f mm，平面深度 %.2f mm，高度 %.2f mm"
            % (
                result["object_points"],
                result["plane_points"],
                result["object_depth_mm"],
                result["plane_depth_mm"],
                result["height_mm"],
            )
        )

    def _update_color(self, image):
        self.color_view.setPixmap(
            QPixmap.fromImage(image).scaled(
                self.color_view.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def _update_depth(self, image):
        self.depth_view.setPixmap(
            QPixmap.fromImage(image).scaled(
                self.depth_view.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def _log(self, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.appendPlainText("[%s] %s" % (timestamp, message))

    def closeEvent(self, event):
        if self.camera is not None:
            self.camera.stop()
        super().closeEvent(event)
