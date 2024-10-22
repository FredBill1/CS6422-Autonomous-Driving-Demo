from typing import Optional, override

import numpy as np
import pyqtgraph as pg

from ..modeling.Car import Car


class CarItem(pg.GraphicsObject):
    LIDAR_BOUNDING_RECT = pg.QtCore.QRectF(-Car.SCAN_RADIUS, -Car.SCAN_RADIUS, 2 * Car.SCAN_RADIUS, 2 * Car.SCAN_RADIUS)

    def __init__(self, car: Optional[Car], *, color=None, with_lidar: bool = False) -> None:
        super().__init__()
        if color is None:
            color = pg.mkQApp().palette().color(pg.QtGui.QPalette.ColorRole.WindowText)
        self._color = color
        self._with_lidar = with_lidar
        self.set_state(car)

    def set_state(self, car: Optional[Car]) -> None:
        self._car = car
        if car is None:
            return

        self.prepareGeometryChange()

        # Outline and wheels geometry
        BOX = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1], [-1, -1]]) / 2
        outline = BOX * [car.LENGTH, car.WIDTH] + [car.LENGTH / 2 - car.BACK_TO_WHEEL, 0]
        wheel = BOX * [car.WHEEL_LENGTH, car.WHEEL_WIDTH]

        # Rotation matrices
        cy, sy = np.cos(car.yaw), np.sin(car.yaw)
        cs, ss = np.cos(car.steer), np.sin(car.steer)
        rot1 = np.array([[cy, -sy], [sy, cy]])
        rot2 = np.array([[cs, -ss], [ss, cs]])

        # Wheel positions
        f_wheel = (rot2 @ wheel.T).T
        fl_wheel = f_wheel + [car.WHEEL_BASE, car.WHEEL_SPACING / 2]
        fr_wheel = f_wheel + [car.WHEEL_BASE, -car.WHEEL_SPACING / 2]
        rl_wheel = wheel + [0, car.WHEEL_SPACING / 2]
        rr_wheel = wheel + [0, -car.WHEEL_SPACING / 2]
        self._items = [outline, fl_wheel, fr_wheel, rl_wheel, rr_wheel]
        for i, x in enumerate(self._items):
            self._items[i] = (rot1 @ x.T).T + [car.x, car.y]
        points = np.concatenate(self._items)
        if self._with_lidar:
            x, y = car.x + cy * Car.BACK_TO_CENTER, car.y + sy * Car.BACK_TO_CENTER
            self._bounding_rect = self.LIDAR_BOUNDING_RECT.translated(x, y)
        else:
            minx, miny, maxx, maxy = points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()
            self._bounding_rect = pg.QtCore.QRectF(minx, miny, maxx - minx, maxy - miny)
        self.update()

    def set_color(self, color: str) -> None:
        self._color = color
        self.update()

    @override
    def paint(self, p: pg.QtGui.QPainter, *args) -> None:
        if self._car is None:
            return
        p.setPen(pg.mkPen(self._color))
        for item in self._items:
            p.drawPolyline(pg.QtGui.QPolygonF([pg.QtCore.QPointF(x, y) for x, y in item]))
        p.drawEllipse(pg.QtCore.QPointF(self._car.x, self._car.y), 0.1, 0.1)
        if self._with_lidar:
            cy, sy = np.cos(self._car.yaw), np.sin(self._car.yaw)
            x, y = self._car.x + cy * Car.BACK_TO_CENTER, self._car.y + sy * Car.BACK_TO_CENTER
            p.drawEllipse(pg.QtCore.QPointF(x, y), Car.SCAN_RADIUS, Car.SCAN_RADIUS)

    @override
    def boundingRect(self) -> pg.QtCore.QRectF:
        return self._bounding_rect if self._car is not None else pg.QtCore.QRectF()
