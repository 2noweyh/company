# Floyd의 최단경로 탐색 알고리즘
# 알고리즘 시간 복잡도 n^3
import copy
def shortest_path_floyd(vertex,W): # 함수
    vsize = len(vertex) #정점 갯수
    D = copy.deepcopy(W) # 깊은 복사

    for k in range(vsize):  # 정점 케이를 추가할떄
        for i in range(vsize):
            for j in range(vsize): # 모든D[i][j] 디 걍신
                if (D[i][k] + D[k][j] < D[i][j]): 
                    D[i][j] = D[i][k] + D[k][j]
        printD(D)  #현재 디를 출력 

def printD(D): #  현재의 퇴단거리 행렬 디를 화면에 출력하는 함수 /  공간복잡도 n^2
    vsize =len(D)
    print('==============================================')
    for i in range(vsize):
        for j in range(vsize):
            if (D[i][j] == INF) : print('IMF', end='')
            else: print('%4d'%D[i][j], end = '')
        print('')


# 편집 거리 (순환)
def edit_distance(S,T,m,n):
    if m == 0 : return n    # S가 공백이면, T의 모든 문자를 S에 삽입
    if n == 0 : return m    # T가 공백이면, S의 모든 문자들을 삭제

    if S[m-1] == T[n-1]:  # 마지막 문자가같으면, 이 문자들 무시
        return edit_distance(S,T,m-1,n-1)
    
# 만약 그렇지 않으면 세 연산을 모두 적용해봄
    return 1 + min(edit_distance(S,T,m,n-1),     # 삽입
                   edit_distance(S,T,m-1,n),     # 대체
                   edit_distance(S,T,m-1,n-1))   # 삭제

# 순환알고리즘은 리턴값으로 바로 내보냄

# 메모제이션 쓰면 반복문을 안쓰고 효율적

# 기반상황과 일반상황을 나눠서 생각할 수 있어야함 

# 아래에 메모제이션 한거 쓰세용 

# 편집 거리 (동적 계획법, 메모이제이션 사용)

def edit_distance_mem(S,T,m,n,mem):
    if m == 0 : return n    # S가 공백이면, T의 모든 문자를 S에 삽입
    if n == 0 : return m    # T가 공백이면, S의 모든 문자들을 삭제

    if mem[m-1][n-1] == None:
        if S[m-1] == T[n-1]:
            mem[m-1][n-1] = edit_distance_mem(S,T,m-1,n-1,mem)
        else: 
            mem[m-1][n-1] = 1 + \
            min(edit_distance_mem(S,T,m,n-1),     # 삽입
                   edit_distance_mem(S,T,m-1,n),     # 대체
                   edit_distance_mem(S,T,m-1,n-1))
            
    return mem[m-1][n-1]


            
            

def find_max(A):
    max = A[0]
    for i in range(len(A)):
        if A[i] > max:
            max = A[i]
    return max

def gcd(a,b):
    while b != 0:
        r = a % b
        a = b
        b = r
    return a

def sequential_search(A, key):
    for i in range(len(A)):
        if A[i] == key:
            return i
    return -1

def compute_aquare_A(n):
    return n*n

def compute_aquare_B(n):
    sum = 0
    for i in range(n):
        sum = sum + n
    return sum

def compute_aquare_C(n):
    sum = 0
    for i in range(n):
        for j in range(n):
            sum = sum + 1
    return sum

def unique_elements(A):
    n = len(A)
    for i in range(n-1):
        for j in range(i+1, n):
            if A[i] == A[j]:
                return False
    return True

def binary_digits(n):
    count = 1
    while n > 1:
        count = count + 1
        n = n //2
    return count

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)


def factorial_for(n):
    ret = 1
    for i in range(1, n+1):
        ret *= i
    return ret


def hanoi_tower(n, fr, tmp, to):
    if(n == 1):
        print("원판 1: %s --> %s" % (fr, to))
    else:               
        hanoi_tower(n-1, fr, to, tmp)
        print("원판 %d: %s --> %s" % (n, fr, to))
        hanoi_tower(n-1,tmp,fr,to)


def selection_sort(A):
    n = len(A)
    for i in range(n-1):
        least = i
        for j in range(i+1, n):
            if A[j] < A[least]:
                least = j
        A[i], A[least] = A[least], A[i]
        print(f'step {i}', A)


def sequential_search(A, key):
    for i in range(len(A)):
        if A[i] == key:
            return i
    return -1





# 문자열 매칭
def string_matcing(T,P):              # 문자열 매칭 함수 정의
    global ans                        # 위치 반환 값을 함수 종료 후에 출력 하기 위해 전역 변수 설정
    n = len(T)                        # 문자열 T의 길이 n값에 대입
    m = len(P)                        # 문자열 P의 길이 m값에 대입
    for i in range(n-m+1):            # 문자열 T에서 첫번째부터 문자열 P의 길이보다 적은 위치까지 탐색
        j = 0                         # j의 초기값 0으로 설정 
        while j<m and P[j] == T[i+j]: # 탐색은 m-1번째 위치까지 하면 됨으로 j는 m보다 작을때까지만 돌리고, 이때 각 위치의 문자열이 같은 지 확인
            j += 1                    # 각 문자열이 같을 시 다음 문자열도 같은지 확인하기 위해 탐색 위치 j에 1을 더함
        if j == m:                    # j의 값이 P의 문자 길이인 m과 같아졌다는 것은 정확히 m개의 값이 같다는 것, 문자열 T 안에서 P를 찾음과 같음
            ans = i                   # 전역 변수 ans에 위치를 대입
            return ans                # 전역 변수 반환 
    ans = -1                          # for문이 다 돌아감에도 함수가 끝이 안났다는것은 문자열을 찾지 못했음을 의미, 전역변수에 -1 을 대입
    return ans   







# closet_pair 함수 복잡도 = n^2 
import math # 거리 구할때 필요한 패키지 불러옴
def distance(p1,p2): # 두 점 사이의 거리 구하는 함수 정의
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2) # 유클리드 거리 정의에 의한 두점 사이의 거리 출력

def closet_pair(p):                     # 가장 인접한 쌍의 거리 구하는 함수 정의
    n = len(p)                          # 2차원 평면상의 점의 갯수 n에 대입
    mindist  = float('inf')             # 시작할때는 최소거리 기준을 무한대로 줌 (2차원 평면 면적 = 무한대)
    # for문을 통해 계속해서 가장 작은 거리를 정의
    for i in range(n-1):                # 비교하는 점 중 첫번째 점은 첫번째 점부터 마지막 점 전까지 매치
        for j in range(i+1, n):         # 비교하는 점 중 두번째 점은 2번째 점부터 마지막 점까지 매치
            dist = distance(p[i],p[j])  # 매치한 비교할 점 첫번째, 두번째 점에 대한 거리 구하기
            if dist < mindist:          # 위에서 구한 거리가 전에 구한 최소거리보다 작은지 비교
                mindist = dist          # 작을 시에는 최소거리 기준을 다시 정의, 아닌 경우는 다음 매치된 점 사이 거리를 구해서 비교
    
    # 모든 for문이 돌아간 후에 정의된 최소거리 즉 가장 근접한 점들 사이의 거리는 mindist 값이 됨
    print(f'평면 P에서 가장 근접한 쌍의 거리는 {mindist}이다') # 가장 근접한 쌍의 거리 출력
                       


def insertion_sort(A):
    n = len(A)
    for i in range(1, n):
        key = A[i]
        j = i - 1
        while j >= 0 and A[j] > key:
            A[j + 1] = A[j]
            j = j - 1
        # A[j + 1] = key
        # printStep(A, i)

def sloe_power(x,n):
    result = 1.0
    for i in range(n):
        result *= x
    return result


def power(x,n):
    if n == 0:
        return 1
    elif n % 2 == 0:
        return power(x*x, n//2)
    else:
        return x*power(x*x, (n-1)//2)
    
n = int(input())
arr = []

for i in range(B):    
	arr.append(list(map(int, input().split())))
        


def partition(A, left, right):
    low = left + 1
    high = right
    pivot = A[left]
    while low <= high:
        while low <= right and A[low] < pivot : low += 1
        while high >= left and A[high] > pivot : high -= 1

        if low < high:
            A[low], A[high] = A[high], A[low]

        A[left], A[high] = A[high], A[left]
        return high
    
def quick_select(A, left, right, k):
    pos = partition(A, left, right)

    if pos+1 == left+k :
        return A[pos]
    elif pos+1 > left+k:
        return quick_select(A, left, pos-1, k)
    else:
        return quick_select(A, pos+1, right, k-(pos+1-left))
        
def quick_select_iter(A, left, right, k):
    while left <= right:
        pivotindex = partition(A, left, right)
        if pivotindex == k-1:
            return A[pivotindex]
        elif pivotindex > k-1:
            right = pivotindex - 1
        else:
            left = pivotindex + 1


# 1번을 해보세요!
def quick_select(A, left, right, k):
    pos = partition(A, left, right)

    if (pos+1 == left+k) :
        return A[pos]
    elif (pos+1 > left+k):
        return quick_select(A, left, pos-1, k)
    else:
        return quick_select(A, pos+1, right, k-(pos+1-left))

# 2번을 해보세요!
def partition(A, left, right):
    low = left + 1
    high = right
    pivot = A[left]
    while low <= high:
        while low <= right and A[low] < pivot : low += 1
        while high >= left and A[high] > pivot : high -= 1

        if low < high:
            A[low], A[high] = A[high], A[low]

        A[left], A[high] = A[high], A[left]
        return high
    



list(tuple(input().split()))

# 24 27 None None None None 45 71 60 9 88 46 38
