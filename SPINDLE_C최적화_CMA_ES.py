import cma
import numpy as np
import time
import matplotlib.pyplot as plt

# --- 1. 가상 환경 설정 (실제로는 SPINDLE 로봇과 연결) ---

def set_spindle_resistance(resistance_c):
    """
    가상의 SPINDLE 로봇에 저항 C값을 설정하는 함수 (시뮬레이션).
    실제 이 함수는 로봇 제어 API를 호출합니다.
    """
    # 저항값은 0과 100 사이로 제한
    safe_c = np.clip(resistance_c, 0, 100)
    print(f"🤖 SPINDLE: 저항을 {safe_c:.1f} (으)로 설정합니다.")
    return safe_c

# --- 2. 뇌 반응 시뮬레이션 (가장 중요한 가상 함수) ---
# 실제 연구에서는 이 함수가 LSL로 EEG/EMG를 받고 MNE로 분석합니다.

def run_virtual_trial(C, add_noise=True):
    C_OPTIMAL = 45.0
    if add_noise:
        noise_level = 0.05 # 잡음 수준
        noise_beta = np.random.randn() * noise_level
        noise_cmc = np.random.randn() * noise_level
        noise_alpha = np.random.randn() * noise_level
        noise_theta = np.random.randn() * noise_level
    else:
        noise_beta, noise_cmc, noise_alpha, noise_theta = 0, 0, 0, 0

    # 1. 분자 계산
    beta_erd = np.exp(-((C - C_OPTIMAL)**2) / (2 * 20**2)) + noise_beta
    cmc = np.exp(-((C - C_OPTIMAL)**2) / (2 * 25**2)) + noise_cmc

    # 2. 분모 계산
    alpha_power_raw = 0.5 * np.exp(-((C - 0)**2) / (2 * 25**2)) + noise_alpha
    theta_power_raw = 0.8 * np.exp(-((C - 100)**2) / (2 * 20**2)) + noise_theta

    # --- 👇 분모 값에 최솟값(baseline) 보장 ---
    min_baseline = 0.05 # 예: 최소 0.05의 값은 가지도록 설정 (0에 가까워지는 것 방지)
    alpha_power = max(alpha_power_raw, min_baseline)
    theta_power = max(theta_power_raw, min_baseline)
    # --- 👆 ---

    # 값 범위 클리핑 (0~1 사이)
    metrics = {
        "beta_erd": np.clip(beta_erd, 0, 1),
        "cmc": np.clip(cmc, 0, 1),
        # 클리핑은 최종 단계에서 수행
        "alpha_power": np.clip(alpha_power, 0, 1),
        "theta_power": np.clip(theta_power, 0, 1),
    }
    return metrics

# --- 3. 최적화 목표 함수 (우리가 정의한 수식) ---

def calculate_objective_function(metrics):
    """
    J(C) 수식을 계산합니다. 0으로 나누는 것을 방지하기 위해 분모에 작은 값(epsilon)을 더합니다.
    """
    # 가중치는 모두 1로 가정 (정규화가 되었다고 전제)
    w1, w2, w3, w4 = 1.0, 1.0, 1.0, 1.0
    epsilon = 1e-6  # 0으로 나누기 방지

    numerator = (w1 * metrics["beta_erd"]) + (w2 * metrics["cmc"])
    denominator = (w3 * metrics["alpha_power"]) + (w4 * metrics["theta_power"]) + epsilon

    return numerator / denominator


# --- 4. 최종 결과 시각화 함수 (새로 추가됨) ---
def plot_final_result(history_c, history_scores, history_best_c, history_best_score):
    """
    최적화가 끝난 후, J(C) 점수 변화 과정을 하나의 그래프로 시각화하여 화면에 보여줍니다.
    """
    all_tested_c = [c for sublist in history_c for c in sublist]
    all_scores = [s for sublist in history_scores for s in sublist]

    plt.figure(figsize=(10, 6)) # 그래프 크기 설정

    # 1. 'Plane' (J(C) 정답) 그리기
    true_c_range = np.linspace(0, 100, 200)
    true_scores = []
    for c_val in true_c_range:
        metrics = run_virtual_trial(c_val, add_noise=False) # 잡음 없이
        score = calculate_objective_function(metrics)
        true_scores.append(score)
    plt.plot(true_c_range, true_scores, 'g--', label="True J(C) 'Plane' (Ground Truth)", linewidth=2)

    # 2. CMA-ES가 시도한 모든 J(C) 점수들
    plt.scatter(all_tested_c, all_scores, c='lightblue', marker='.', label='All Tested C values & Scores', alpha=0.5)

    # 3. 각 세대별 '최고 J(C) 점수'의 변화 과정
    # 최고 점수가 갱신될 때마다 빨간 점과 선으로 표시
    plt.plot(history_best_c, history_best_score, 'r.-', label='Best Score per Generation', markersize=10, linewidth=1.5)
    # 최종 최적 지점 강조
    plt.scatter(history_best_c[-1], history_best_score[-1], c='red', marker='*', s=300, label=f'Final Best C: {history_best_c[-1]:.2f}', zorder=5, edgecolors='black')

    plt.title("CMA-ES Optimization History for Optimal SPINDLE Resistance")
    plt.xlabel("SPINDLE Resistance (C)")
    plt.ylabel("Objective Score J(C) - Brain Optimization State")
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.xlim(0, 100)

    # 그래프를 파일 저장이 아닌 화면에 바로 띄움
    plt.show()

# --- 5. CMA-ES 최적화 메인 루프 ---

def main_optimization_loop():
    
    print("🧠 SPINDLE 뇌 최적화 시뮬레이션을 시작합니다.")
    print("CMA-ES 알고리즘이 환자의 '최적 저항(C)'을 탐색합니다...\n")

    # CMA-ES 초기 설정
    C_initial_guess = 30.0  # 초기 추측값 (아무 값이나 상관없음)
    C_sigma = 20.0          # 초기 탐색 범위 (0~100 사이에서 20정도의 범위)
    bounds = [0, 100]       # 저항 C의 물리적 한계 (0 ~ 100)
    
    # CMA-ES 객체 생성
    # [C_initial_guess]는 1차원 문제(C 하나만 찾음)라는 의미
    es = cma.CMAEvolutionStrategy([C_initial_guess], C_sigma, {'bounds': bounds})
    
    n_generations = 10  # 총 10 세대(탐색 단계) 동안 최적화

    # 시각화를 위한 데이터 저장 리스트
    history_c = []          # 각 세대에서 시도된 C 값들
    history_scores = []     # 각 세대에서 얻은 점수들
    history_best_c = []     # 각 세대까지의 최고 C 값
    history_best_score = [] # 각 세대까지의 최고 점수

    current_best_c =C_initial_guess
    current_best_score = -np.inf
    
    # CMA-ES는 기본적으로 "최소화"를 하므로, 우리는 점수에 -1을 곱해 "최대화" 문제로 바꿉니다.
    
    for gen in range(n_generations):
        print(f"\n--- 🚀 세대 {gen + 1}/{n_generations} ---")
        
        # 1. Ask: CMA-ES에게 다음으로 시도해 볼 C값 후보들을 물어봅니다.
        c_solutions_arrays = es.ask()
        c_solutions = [arr[0] for arr in c_solutions_arrays]
        scores = []
        
        gen_c_values = [] #현재 세대의 C값들 저장용

        # 2. Evaluate: 추천받은 C값들을 하나씩 시뮬레이션(실험)합니다.
        for i, C in enumerate(c_solutions):
            
            # 2a. 로봇에 저항 저장
            safe_c = set_spindle_resistance(C)
            gen_c_values.append(safe_c) #시각화를 위해 저장
            
            
            # 2b. 가상 환자가 해당 저항으로 동작 수행 (실제로는 LSL/MNE 분석)
            print(f"  [시도 {i+1}] 환자가 {safe_c:.1f} 저항으로 동작 수행 중...")
            time.sleep(0.5)  # 1회 동작에 0.5초 걸린다고 가정
            
            # 2c. 뇌/근육 지표 수집
            metrics = run_virtual_trial(safe_c, add_noise=True)
            
            # 2d. 최종 점수 계산
            score = calculate_objective_function(metrics)
            scores.append(score)
            
            print(f"  [결과 {i+1}] J(C) 점수: {score:.4f}")
            print(f"      (운동: {metrics['beta_erd']:.2f}, 연결: {metrics['cmc']:.2f} | 지루함: {metrics['alpha_power']:.2f}, 과부하: {metrics['theta_power']:.2f})")

        # 3. Tell: 각 C값과 그에 대한 점수(-1 곱해서)를 CMA-ES에게 알려줍니다.
        # 알고리즘은 이 정보를 바탕으로 다음 세대에 탐색할 '더 좋은 동네'를 학습합니다.
        es.tell(c_solutions_arrays, [-s for s in scores])
        
        # 현재까지의 최고 점수와 C값 출력
        es.disp()

        # 현재 세대의 최고 점수 및 C값 업데이트
        gen_best_score_index = np.argmax(scores)
        gen_best_score = scores[gen_best_score_index]
        if gen_best_score > current_best_score:
            current_best_score = gen_best_score
            current_best_c = gen_c_values[gen_best_score_index]

        # 히스토리 저장
        history_c.append(gen_c_values)
        history_scores.append(scores)
        history_best_c.append(current_best_c)
        history_best_score.append(current_best_score)

# --- 최종 결과 출력 ---
    print("\n--- ✅ 최적화 종료 ---")
    final_best_c = es.result.xbest[0]
    final_best_score = -es.result.fbest
    print(f"\n최종 추천 저항 C 값: {final_best_c:.2f}")
    print(f"예상되는 최적 점수 J(C): {final_best_score:.4f}")
    print("(시뮬레이션의 실제 최적값은 45.0 이었습니다.)")

    # --- 최종 시각화 ---
    print("\n📊 최적화 결과 그래프를 보여줍니다...")
    plot_final_result(history_c, history_scores, history_best_c, history_best_score)

# --- 5. 스크립트 실행 ---
if __name__ == "__main__":
    main_optimization_loop()