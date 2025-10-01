import numpy as np
import matplotlib.pyplot as plt

def fquad(x):
    """최적화할 목적 함수 예시 (타원형 2차 함수)"""
    #최적점은 [3,5]
    f = (x[0] - 3)**2 + 10*(x[1] - 5)**2
    return f

def fackley(x, a=20, b=0.2, c=2*np.pi):
    """
    에크리 함수 (Ackley Function)
    넓은 그릇 모양 안에 수많은 지역 최적점이 존재.
    전역 최적점은 [0, 0]에서 f=0으로 유일하다.
    """
    N = len(x)
    term1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / N))
    term2 = -np.exp(np.sum(np.cos(c * x)) / N)
    return term1 + term2 + a + np.exp(1)

def purecmaes_2d():
    #----------------------------------------1. 초기설정: 2차원 문제 정의 -------------------------------------
    strfitnessfct = fquad # or fackley
    N = 2
    xmean = np.array([[8.0],[8.0]]) #시작점 (열벡터) basecamp
    sigma = 3.0 #초기 보폭
    stopfitness = 1e-10
    stopeval = 1e3*N**2

    #----------------------------------------2. 전략 파라미터 설정 (N=2에 맞춰 자동계산)-------------------------
    lambda_ = 4 + int(3*np.log(N)) #한 세대의 대원 수 로그적으로 완만하게 증가 -> 효율적 탐색
    mu = lambda_//2 #대원 절반이 우수 대원 수
    weights = np.log(mu + 0.5) - np.log(np.arange(1,mu+1)) #가중치: 우수대원 중 1등을 가장 많이 mu를 가장 적게 반영
    weights = weights/np.sum(weights)# 가중치 정규화
    weights = weights.reshape(mu,1) #다른 행렬과 곱하기 적합한 형태로 만들기 위해서
    mueff = np.sum(weights)**2/np.sum(weights**2) #분산 유효 선택 질량 ---> 실질적 다음세대에 영향 미칠 해답 개수

    cc = (4 + mueff/N) / (N + 4 + 2*mueff/N) #pc 의 망각률을 정한다. 
    cs = (mueff + 2)/(N+mueff+5)#ps의 망각률
    c1 = 2/((N+1.3)**2 + mueff) # 성공적인 이동경로 pc를 얼마나 반영할 것인지 결정
    cmu = min(1-c1, 2*(mueff-2+1/mueff)/((N+2)**2+mueff)) #개별 대원들의 분포를 얼마나 반영할 것인지 결정
    damps = 1+2*max(0,np.sqrt((mueff-1)/(N+1))-1) +cs #보폭이 너무 급격하게 커지거나 작아지는것을 막아준다. 
    #-----------------------------------------3. 동적 파라미터 초기화 ------------------------------------------
    pc = np.zeros((N,1)) # 진화 경로: 어떤 방향으로 의미 있는 성공적인 이동을 했는지 기록하는 전략노트--> 초기화 다 0으로 채움
    ps = np.zeros((N,1)) # 보폭을 업데이트하는데 사용되는 진화 경로 마찬가지로 0으로 초기화
    B = np.eye(N) #변수명: Basis , C를 구성하는 요소중 하나로 좌표축 방향 정의, np.eye(N) 는 N X N 크기 단위행렬 생성
    D = np.ones((N,1)) #변수명: Diagonal , C를 구성하는 또 다른 요소 좌표축 길이 정의 np.ones((N,1)) 는 N행 1열짜리 1로 채워진 열백터 생성
    C = B @ np.diag(D.flatten()**2)@B.T #BΛB^T --> B: 모든 고유벡터를 열로 모아놓은 행렬 주축 방향, Λ: 각 축으로 얼마나 퍼져있는지
    invsqrtC = B @ np.diag(D.flatten()**-1)@B.T #C**-1
    eigeneval = 0
    chiN = N**0.5*(1-1/(4*N)+1/(21*N**2)) #(E∥N(0,I)∥) --> ps 의 실제 길이가 chiN보다 길면 보폭을 늘리고 , 짧으면 줄인다.

    #시각화를 위한 데이터 저장 리스트
    history = []

    #-----------------------------------------4. 메인 반복문 (Generation Loop)---------------------------------
    counteval = 0
    while counteval < stopeval:
        counteval_generation = counteval #현재 세대 시작 시점 기록
        history.append(xmean.copy()) #경로 시각화를 위해 현재 평균 위치 저장

        #4-1. 새로운 자손 생성 및 평가
        arx = np.zeros((N,lambda_)) #여기에 lambda_명의 자손들의 N차원 좌표가 저장될거임
        arfitness = np.zeros(lambda_) #lambda_ 길이의 0으로 채워진 1차원 배열. 각 자손의 점수 저장
        for k in range(lambda_): # k 는 0~lambda_-1
            sample = np.random.randn(N,1) #탐사의 기본재료인 무작위 방향백터
            arx[:,k] = (xmean + sigma*B@(D*sample)).flatten()
            # D*sample : 무작위방향을 탐색 모양 축 길이 맞춘다. (scaling)
            # B @ (...) : 스케일링 된 탐색 방향을 현재 학습된 최적 방향으로 회전
            # sigma * ... : 회전된 탐색 방향에 현재 보폭(탐색 반경)을 곱해 실제 이동 거리를 정함.
            # xmean + ... : 계산된 이동벡터를 현재 베이스캠프위치에 더해 최종 위치를 정함
            # .flatten() : 계산된 N행 1열짜리 열벡터를 1차원 배열로 바꿔서 arx의 k번째 열에 저장하기 쉽게 만듦
            arfitness[k] = strfitnessfct(arx[:,k]) #방금 생성한 후보지점의 점수 계산
            counteval+=1

        #4-2. 정렬 및 베이스캠프(평균) 이동
        arindex = np.argsort(arfitness) # 점수를 낮은순으로 인덱스 저장한 리스트 전달
        xold = xmean
        xmean = arx[:,arindex[:mu]] @ weights
          # arx[:,arindex[:mu]] :  전체 후보지점의 좌표가 담긴 arx 행렬에서 위에서 우수대원의 좌표반환
          # ...@ weights : 우수 대원들 좌표와 가중치를 행렬곱셈 --> 가중치를 곱해서 가중평균 계산하는 과정
          # 가중 평균이 xmean이 된다. 

        #4-3. 진화 경로 업데이트
        ps = (1-cs) * ps + np.sqrt(cs * (2-cs) * mueff)*invsqrtC @ (xmean - xold)/sigma
          # (xmean - xold) : xmean과 xold의 차이
          # (xmean - xold)/sigma : 이동 벡터/현재보폭 -> 정규화 -> 보폭의 크기와 상관없이 상대적이동량
          # invsqrtC @ (...) : 좌표 변환 ->invsqrtC는 공부산행렬C의 역행렬의 제곱근이고 이 행렬을 곱함으로써 보폭으로 정규화된 이동백터를 현재의 탐색공간 C를 고려한 기준공간(구형)
          # np.sqrt(cs*(2-cs)*mueff) : 학습률과 가중치, 새로운 정보를 얼마나 빠르게 받아드리고 반영할 것인가?
          # (1-cs) * ps : 이전 세대의 ps 값에 기억률을 곱해서 과거의 ps정보를 일정 비율로 반영
        hsig_cond = np.linalg.norm(ps) / np.sqrt(1-(1-cs)**(2*(counteval-counteval_generation)/lambda_))/chiN
          # hsig_cond는 보폭 경로 ps의 길이가 예상되는 무작위 이동길이에 비해 얼마나 긴지 평가하는 조건 값
          # ps의 길이가 chiN(표준정규분포를 따르는 벡터 길이 기대값)의 길이보다 훨씬 길다면 의미있는 진전이라고 판단
        hsig = 1 if hsig_cond<1.4 + 2/ (N+1) else 0 #C 업데이트시 안정성 확보 - 현재 이동 굳=1 , 현재 이동 벧=0
        pc = (1-cc)*pc + hsig * np.sqrt(cc * (2-cc) * mueff) *(xmean-xold)/sigma #공분산행렬(C) 조절을 위한 진화 경로 pc 업데이트

        # 4-4. 공분산 행렬 C 업데이트 
        artmp = (1/sigma)*(arx[:,arindex[:mu]] - xold) #우수한 평가를 받은 자손들이 이전 베이스 캠프로부터 어느 방향으로 얼마나 이동했는지 나타내는 행렬
        C = (1-c1-cmu)* C + c1 * (pc@pc.T + (1-hsig)*cc*(2-cc)*C)+cmu*artmp@np.diag(weights.flatten())@artmp.T 

        #4-5. 보폭 sigma 업데이트
        sigma = sigma * np.exp((cs/damps)*(np.linalg.norm(ps)/chiN - 1))

        #4-6. C 행렬 분해 (주기적으로 수행)
        if counteval - eigeneval > lambda_ / (c1 + cmu) / N /10: #C행렬이 충분히 업데이트 되어 변화가 누적되었을때만 고유값 분해를 수행하도록 해서 계산 효율성을 높인다. 
            eigeneval = counteval #고유값 분해를 수행했으므로, eigeneval 값을 현재의 총 함수 평가 횟수 counteval로 업데이트
            C = np.triu(C) + np.triu(C,1).T 
            # np. triu(C) : C 행렬의 상삼각부분(대각선 포함)만 남기고 하삼각 분류를 0으로 만듦
            # np. triu(C,1).T : C 행렬의 엄밀한 상삼각부분(대각선 제외)만 남긴 후 전치(Transpose)해서 하삼각부분 만든다
            # 이 둘을 더해서 C를 이론에 가까운 대칭 행렬로 만든다.


            eigenvalues, B = np.linalg.eig(C)
            #고유값이 음수가 되는 수치적 오류 방지
            if np.any(eigenvalues<=0):
                eigenvalues[eigenvalues<=0] = 1e-12
            D = np.sqrt(eigenvalues).reshape(N,1) 
            invsqrtC = B @ np.diag(D.flatten()**-1)@B.T
        
        #4-7. 종료 조건 확인
        if arfitness[arindex[0]] <= stopfitness:
            break
    history.append(xmean.copy())
    xmin = arx[:, arindex[0]] #최종적으로 가장 좋은 해답 저장

    return xmin, np.array(history)

def plot_cmaes_path(history, fitness_func, x_range, y_range, global_minimum):
    """
    CMA-ES의 탐색 경로를 동적으로 시각화하는 함수
    - history: 탐색 경로 데이터
    - fitness_func: 시각화할 목적 함수 객체
    - x_range, y_range: 시각화할 x, y 좌표 범위 (예: [-5, 5])
    - global_minimum: 함수의 전역 최적점 좌표 (예: [0, 0])
    """
    path = history.squeeze().T
    
    x = np.linspace(x_range[0], x_range[1], 400)
    y = np.linspace(y_range[0], y_range[1], 400)
    X, Y = np.meshgrid(x, y)
    
    # Z 계산 부분을 전달받은 함수로 동적으로 변경
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = fitness_func(np.array([X[i, j], Y[i, j]]))
    
    plt.figure(figsize=(10, 8))
    # levels를 자동으로 설정하여 어떤 함수든 잘 보이게 함
    plt.contour(X, Y, Z, levels=50, cmap='viridis_r')
    
    # 경로 데이터 그리기
    if path.shape[1] > 1:
        plt.plot(path[0, :], path[1, :], '-o', color='red', markersize=4, linewidth=2, label='Path of Mean (m)')
    else:
        plt.plot(path[0], path[1], 'ro', markersize=4, label='Path of Mean (m)')

    # 전역 최적점 위치를 동적으로 받아서 표시
    if global_minimum is not None:
        plt.plot(global_minimum[0], global_minimum[1], 'y*', markersize=20, label='Global Minimum')
        
    plt.title(f'CMA-ES Search Path ({fitness_func.__name__})') # 함수 이름도 제목에 표시
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# 스크립트 실행
if __name__ == '__main__':
    # purecmaes_2d 함수 내부에서 strfitnessfct = fquad 로 설정했다고 가정
    best_solution, history_data = purecmaes_2d()
    print("Best solution found:")
    print(best_solution)
    
    # 탐색 경로 시각화 시, fquad 함수 정보와 범위를 전달
    plot_cmaes_path(
        history=history_data, 
        fitness_func=fquad, 
        x_range=[-2, 12], 
        y_range=[-2, 12], 
        global_minimum=[3, 5]
    )

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# 스크립트 실행
if __name__ == '__main__':
    # purecmaes_2d 함수 내부에서 strfitnessfct = fackley 로 설정했다고 가정
    best_solution, history_data = purecmaes_2d()
    print("Best solution found:")
    print(best_solution)
    
    # 탐색 경로 시각화 시, fackley 함수 정보와 범위를 전달
    plot_cmaes_path(
        history=history_data, 
        fitness_func=fackley, 
        x_range=[-5, 5], 
        y_range=[-5, 5], 
        global_minimum=[0, 0]
    )
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""