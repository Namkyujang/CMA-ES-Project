import numpy as np
import time
import matplotlib.pyplot as plt
from skopt import gp_minimize # <-- 베이즈 최적화 (Gaussian Process Minimization)
from skopt.space import Real # <-- 탐색 범위 정의
from skopt.utils import use_named_args # <-- 함수 인자 매핑

# --- 1. 가상 환경 설정  ---
def set_spindle_resistance(resistance_c):
    safe_c = np.clip(resistance_c, 0, 100)
    print(f"🤖 SPINDLE: 저항을 {safe_c:.1f} (으)로 설정합니다.")
    return safe_c

# --- 2. 뇌 반응 시뮬레이션 ---
def run_virtual_trial(C, add_noise=True):
    C_OPTIMAL = 45.0
    if add_noise:
        noise_level = 0.05
        noise_beta = np.random.randn() * noise_level
        noise_cmc = np.random.randn() * noise_level
        noise_alpha = np.random.randn() * noise_level
        noise_theta = np.random.randn() * noise_level
    else:
        noise_beta, noise_cmc, noise_alpha, noise_theta = 0, 0, 0, 0

    beta_erd = np.exp(-((C - C_OPTIMAL)**2) / (2 * 20**2)) + noise_beta
    cmc = np.exp(-((C - C_OPTIMAL)**2) / (2 * 25**2)) + noise_cmc
    min_baseline = 0.05
    alpha_power_raw = 0.5 * np.exp(-((C - 0)**2) / (2 * 25**2)) + noise_alpha
    theta_power_raw = 0.8 * np.exp(-((C - 100)**2) / (2 * 20**2)) + noise_theta
    alpha_power = max(alpha_power_raw, min_baseline)
    theta_power = max(theta_power_raw, min_baseline)

    metrics = {
        "beta_erd": np.clip(beta_erd, 0, 1),
        "cmc": np.clip(cmc, 0, 1),
        "alpha_power": np.clip(alpha_power, 0, 1),
        "theta_power": np.clip(theta_power, 0, 1),
    }
    return metrics

# --- 3. 최적화 목표 함수 J(C) (그대로 유지) ---
def calculate_objective_function(metrics):
    w1, w2, w3, w4 = 1.0, 1.0, 1.0, 1.0
    epsilon = 1e-6
    numerator = (w1 * metrics["beta_erd"]) + (w2 * metrics["cmc"])
    denominator = (w3 * metrics["alpha_power"]) + (w4 * metrics["theta_power"]) + epsilon
    return numerator / denominator

# --- 4. 베이즈 최적화를 위한 Wrapper 함수 ---
# BO는 '최소화'를 목표로 하므로, J(C) 점수에 -1을 곱해서 반환합니다.

# 탐색 공간 정의 (C는 0에서 100 사이의 실수)
search_space = [Real(0.0, 100.0, name='C')]

# gp_minimize가 호출할 함수
@use_named_args(search_space)
def objective_for_bo(**params):
    C = params['C']

    # 1. 로봇 설정
    safe_c = set_spindle_resistance(C)

    # 2. 실험 수행 (시뮬레이션)
    print(f"  [시도] 환자가 {safe_c:.1f} 저항으로 동작 수행 중...")
    time.sleep(0.5) # 실제 실험 시간 가정

    # 3. 결과 측정
    metrics = run_virtual_trial(safe_c, add_noise=True)

    # 4. 점수 계산
    score = calculate_objective_function(metrics)
    print(f"  [결과] J(C) 점수: {score:.4f}")
    print(f"      (운동: {metrics['beta_erd']:.2f}, 연결: {metrics['cmc']:.2f} | 지루함: {metrics['alpha_power']:.2f}, 과부하: {metrics['theta_power']:.2f})")

    # 5. BO는 최소화를 하므로, 점수의 음수값을 반환
    return -score

# --- 5. 최종 결과 시각화 함수 (BO 결과에 맞게 수정) ---
def plot_bo_result(result):
    """
    BO 최적화 결과를 시각화합니다.
    """
    evaluated_c_values = [x[0] for x in result.x_iters] # BO가 시도한 모든 C값들
    obtained_scores = -result.func_vals # 얻은 점수들 (-1 곱해서 원래 점수로)

    best_c_index = np.argmin(result.func_vals) # 가장 낮은 func_val (가장 높은 score)의 인덱스
    best_c = evaluated_c_values[best_c_index]
    best_score = obtained_scores[best_c_index]

    plt.figure(figsize=(10, 6))

    # 1. '진짜 Plane' 그리기
    true_c_range = np.linspace(0, 100, 200)
    true_scores = []
    for c_val in true_c_range:
        metrics = run_virtual_trial(c_val, add_noise=False)
        score = calculate_objective_function(metrics)
        true_scores.append(score)
    plt.plot(true_c_range, true_scores, 'g--', label="True J(C) 'Plane' (Ground Truth)", linewidth=2)

    # Y축 상한선 계산
    reasonable_y_max = max(max(true_scores) * 1.1, 10)

    # 2. BO가 시도한 모든 점수 (상한선 적용)
    clipped_scores = np.clip(obtained_scores, 0, reasonable_y_max)
    plt.scatter(evaluated_c_values, clipped_scores, c='lightblue', marker='.', label='All Tested C values & Scores (Clipped)', alpha=0.7, s=50)

    # 3. BO가 찾은 최종 최적 지점 강조
    final_best_clipped = np.clip(best_score, 0, reasonable_y_max)
    plt.scatter(best_c, final_best_clipped, c='red', marker='*', s=300, label=f'Final Best C (BO): {best_c:.2f}', zorder=5, edgecolors='black')

    plt.title("Bayesian Optimization History for Optimal SPINDLE Resistance")
    plt.xlabel("SPINDLE Resistance (C)")
    plt.ylabel("Objective Score J(C) - Brain Optimization State")
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.ylim(0, reasonable_y_max)
    plt.xlim(0, 100)

    plt.show()


# --- 6. 베이즈 최적화 메인 루프 ---
def main_bayesian_optimization():
    print("🧠 SPINDLE 뇌 최적화 (베이즈 최적화 방식) 시뮬레이션을 시작합니다.")
    print("알고리즘이 환자의 '최적 저항(C)'을 탐색합니다...\n")

    n_calls = 20 # 총 20번의 실험(평가) 수행 (CMA-ES 10세대 * 4회 = 40회보다 적음)
    n_initial_points = 5 # 처음 5번은 무작위 탐색

    # gp_minimize 함수 호출
    result = gp_minimize(
        func=objective_for_bo,       # 최소화할 목적 함수 (-J(C))
        dimensions=search_space,     # 탐색 공간 (C: 0~100)
        n_calls=n_calls,             # 총 함수 호출(실험) 횟수
        n_initial_points=n_initial_points, # 초기 무작위 탐색 횟수
        acq_func='EI',               # 다음 탐색 지점 결정 전략 (Expected Improvement)
        random_state=123             # 재현성을 위한 시드
    )

    # --- 최종 결과 출력 ---
    print("\n--- ✅ 최적화 종료 ---")
    final_best_c = result.x[0]             # 찾은 최적의 C값
    final_best_score = -result.fun         # 찾은 최적 점수 (-1 곱해서 원래 점수로)

    print(f"\n최종 추천 저항 C 값: {final_best_c:.2f}")
    print(f"예상되는 최적 점수 J(C): {final_best_score:.4f}")
    print("(시뮬레이션의 실제 최적값은 45.0 이었습니다.)")

    # --- 최종 시각화 ---
    print("\n📊 최적화 결과 그래프를 보여줍니다...")
    plot_bo_result(result)

# --- 7. 스크립트 실행 ---
if __name__ == "__main__":
    main_bayesian_optimization()