// Fit the bounding sphere in both dimensions, including portrait viewports.
export function perspectiveFitDistance(radius, verticalFovDegrees, aspect, padding = 1.28) {
  if (![radius, verticalFovDegrees, aspect, padding].every(Number.isFinite) ||
      radius <= 0 || verticalFovDegrees <= 0 || verticalFovDegrees >= 180 || aspect <= 0 || padding < 1) {
    throw new Error("自动取景需要有效的半径、视场角、画幅比例和边距。");
  }
  const verticalFov = verticalFovDegrees * Math.PI / 180;
  const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * aspect);
  return radius / Math.sin(Math.min(verticalFov, horizontalFov) / 2) * padding;
}
