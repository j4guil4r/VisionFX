import cv2
import numpy as np
import pytest

from app_ui.landmark_detector import LandmarkDetector

# Modelo 3D canónico correspondiente a los landmarks faciales de MediaPipe (mm)
CANONICAL_FACE_3D = np.array([
    [0.0, 0.0, 0.0],          # 1:   Punta de la nariz
    [0.0, 110.0, 65.0],       # 199: Mentón
    [-75.0, -60.0, 65.0],     # 33:  Comisura ojo izquierdo
    [75.0, -60.0, 65.0],      # 263: Comisura ojo derecho
    [-40.0, 50.0, 35.0],      # 61:  Comisura boca izquierda
    [40.0, 50.0, 35.0]        # 291: Comisura boca derecha
], dtype=np.float64)


def create_detector_instance():
    """Instancia LandmarkDetector aislando el resolver de MediaPipe."""
    obj = LandmarkDetector.__new__(LandmarkDetector)
    obj.FACE_3D = CANONICAL_FACE_3D.copy()
    obj.landmark_indices = [1, 199, 33, 263, 61, 291]
    return obj


@pytest.fixture
def detector():
    return create_detector_instance()


@pytest.fixture
def camera_intrinsics():
    w, h = 1280.0, 720.0
    cam_matrix = np.array([
        [w, 0.0, w / 2.0],
        [0.0, w, h / 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    return cam_matrix, dist_coeffs


def test_rotation_matrix_orthogonality_and_determinant():
    """Valida que la matriz R derivada de rvec vía Rodrigues sea estrictamente ortogonal y pertenezca a SO(3)."""
    np.random.seed(123)
    # Probar múltiples orientaciones aleatorias en rango [-pi, pi]
    random_vectors = np.random.uniform(-np.pi, np.pi, size=(25, 3))

    for angles in random_vectors:
        rvec = angles.astype(np.float64)
        rot_mat, _ = cv2.Rodrigues(rvec)

        # Dimensiones estrictas 3x3
        assert rot_mat.shape == (3, 3)

        # Ortogonalidad: R * R^T = I y R^T * R = I
        identity = np.eye(3, dtype=np.float64)
        np.testing.assert_allclose(rot_mat @ rot_mat.T, identity, atol=1e-6)
        np.testing.assert_allclose(rot_mat.T @ rot_mat, identity, atol=1e-6)

        # Determinante unitario det(R) = +1 (rotación propia sin reflexión)
        det_r = np.linalg.det(rot_mat)
        assert np.isclose(det_r, 1.0, atol=1e-6)


def test_ground_truth_reconstruction_consistency(detector, camera_intrinsics):
    """Valida la consistencia de solvePnP al reconstruir pose sintética exacta."""
    cam_matrix, dist_coeffs = camera_intrinsics
    rvec_gt = np.array([0.15, -0.20, 0.05], dtype=np.float64)
    tvec_gt = np.array([10.0, -15.0, 650.0], dtype=np.float64)

    # Proyección 3D -> 2D
    proj_pts, _ = cv2.projectPoints(CANONICAL_FACE_3D, rvec_gt, tvec_gt, cam_matrix, dist_coeffs)
    points_2d = [tuple(pt[0]) for pt in proj_pts]

    pose = detector.estimate_pose(points_2d, cam_matrix, dist_coeffs)
    assert pose is not None, "estimate_pose no logró converger con datos sintéticos válidos"

    rvec_est, tvec_est = pose
    np.testing.assert_allclose(rvec_est.ravel(), rvec_gt, atol=1e-3)
    np.testing.assert_allclose(tvec_est.ravel(), tvec_gt, atol=1e-2)


def test_translation_vector_valid_ranges(detector, camera_intrinsics):
    """Verifica que el vector de traslación se mantenga en rangos geométricos válidos."""
    cam_matrix, dist_coeffs = camera_intrinsics
    rvec_gt = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    tvec_gt = np.array([0.0, 0.0, 500.0], dtype=np.float64)

    proj_pts, _ = cv2.projectPoints(CANONICAL_FACE_3D, rvec_gt, tvec_gt, cam_matrix, dist_coeffs)
    points_2d = [tuple(pt[0]) for pt in proj_pts]

    pose = detector.estimate_pose(points_2d, cam_matrix, dist_coeffs)
    assert pose is not None

    _, tvec = pose
    tvec_flat = tvec.ravel()

    # Valores finitos
    assert np.all(np.isfinite(tvec_flat)), "tvec contiene valores no numéricos o infinitos"

    # Profundidad Z estrictamente positiva (delante de la cámara)
    assert tvec_flat[2] > 0.0, "La profundidad Z debe ser positiva"

    # Rango físicamente coherente en mm para tracking facial
    assert 100.0 <= tvec_flat[2] <= 3000.0, f"Profundidad Z fuera de rango: {tvec_flat[2]}"


def test_null_and_empty_inputs_handling(detector, camera_intrinsics):
    """Verifica robustez ante entradas nulas o vacías."""
    cam_matrix, dist_coeffs = camera_intrinsics

    assert detector.estimate_pose(None, cam_matrix, dist_coeffs) is None
    assert detector.estimate_pose([], cam_matrix, dist_coeffs) is None


def test_noisy_inputs_stability(detector, camera_intrinsics):
    """Valida estabilidad numérica y preservación de SO(3) bajo perturbación gaussiana."""
    cam_matrix, dist_coeffs = camera_intrinsics
    rvec_gt = np.array([0.1, -0.1, 0.0], dtype=np.float64)
    tvec_gt = np.array([0.0, 0.0, 600.0], dtype=np.float64)

    proj_pts, _ = cv2.projectPoints(CANONICAL_FACE_3D, rvec_gt, tvec_gt, cam_matrix, dist_coeffs)
    clean_2d = proj_pts.reshape(-1, 2)

    # Inyección de ruido gaussiano (sigma = 1.5 px)
    np.random.seed(42)
    noise = np.random.normal(loc=0.0, scale=1.5, size=clean_2d.shape)
    noisy_2d = clean_2d + noise
    points_2d = [tuple(pt) for pt in noisy_2d]

    pose = detector.estimate_pose(points_2d, cam_matrix, dist_coeffs)
    assert pose is not None, "solvePnP falló bajo ruido moderado"

    rvec_est, tvec_est = pose
    rot_mat, _ = cv2.Rodrigues(rvec_est)

    # Ortogonalidad preservada
    np.testing.assert_allclose(rot_mat @ rot_mat.T, np.eye(3), atol=1e-5)
    assert np.isclose(np.linalg.det(rot_mat), 1.0, atol=1e-5)

    # Desviación de traslación acotada
    tvec_diff = np.linalg.norm(tvec_est.ravel() - tvec_gt)
    assert tvec_diff < 50.0, f"Divergencia excesiva en tvec con ruido: error={tvec_diff:.2f}mm"


def test_degenerate_inputs_handling(detector, camera_intrinsics):
    """Verifica manejo seguro ante geometrías degeneradas o valores inválidos."""
    cam_matrix, dist_coeffs = camera_intrinsics

    # Puntos colapsados en un único píxel
    degenerate_pts = [(300.0, 300.0)] * 6
    try:
        pose = detector.estimate_pose(degenerate_pts, cam_matrix, dist_coeffs)
        if pose is not None:
            _, tvec = pose
            assert np.all(np.isfinite(tvec))
    except cv2.error:
        pass

    # Entrada con NaN
    nan_pts = [(np.nan, np.nan)] * 6
    try:
        pose = detector.estimate_pose(nan_pts, cam_matrix, dist_coeffs)
        assert pose is None or not np.any(np.isnan(pose[1]))
    except (cv2.error, ValueError):
        pass
