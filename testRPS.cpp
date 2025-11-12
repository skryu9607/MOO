#include <iostream>
#include <vector>
#include <map>
#include <functional> // std::function
#include <stdexcept>  // std::invalid_argument
#include <numeric>    // std::accumulate
#include <iomanip>    // std::setprecision
#include <any>        // std::any (솔루션 데이터를 유연하게 저장하기 위해)
#include <string>
#include <algorithm>  // std::find_if

// 가중치 벡터(w)를 간단히 타입으로 정의
using WeightVector = std::vector<double>;
// Neighborhood는 꼭짓점(가중치 벡터)들의 집합(simplex)
using Neighborhood = std::vector<WeightVector>;

/**
 * @brief 사용자의 스칼라화된 문제 솔버가 반환해야 하는 결과 구조체
 */
struct SolutionResult {
    // 실제 솔루션 데이터 (예: 경로, 로봇 설정 등).
    // 어떤 타입이든 저장할 수 있도록 std::any 사용.
    std::any solution_data; 
    
    // 해당 솔루션의 목적 함수 값 벡터 (f(s*))
    std::vector<double> objective_vector;
};

/**
 * @brief 최대 후회(regret)를 찾는 LP 솔버의 결과 구조체
 */
struct MaxRegretResult {
    WeightVector w_star;                // 찾은 최적의 가중치 벡터
    double max_regret = -1.0;           // 해당 가중치의 후회 값 (상한)
    Neighborhood neighborhood_to_split; // w*가 속한 Neighborhood
    bool found = false;                 // 성공적으로 찾았는지 여부
};


// --- 전역 헬퍼 함수 (벡터 출력을 위해) ---
void print_vec(const WeightVector& vec) {
    std::cout << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << std::fixed << std::setprecision(3) << vec[i];
        if (i < vec.size() - 1) std::cout << ", ";
    }
    std::cout << "]";
}


class MRPS {
public:
    // 사용자가 제공할 솔버 함수의 시그니처 정의
    // const WeightVector& 를 받아 SolutionResult 를 반환
    using SolverFunction = std::function<SolutionResult(const WeightVector&)>;

    /**
     * @brief MRPS 알고리즘 생성자
     * @param num_objectives 목적 함수의 개수 (n)
     * @param solver_func 가중치 벡터(w)를 받아 스칼라화된 문제를 푸는 함수
     */
    MRPS(int num_objectives, SolverFunction solver_func)
        : num_objectives_(num_objectives), solve_problem_func_(std::move(solver_func)) {
        if (num_objectives_ < 2) {
            throw std::invalid_argument("Must have at least 2 objectives.");
        }
    }

    /**
     * @brief MRPS 알고리즘 실행 (Algorithm 1)
     * @param K 총 샘플링할 가중치의 개수
     */
    void run(int K) {
        if (K < num_objectives_) {
            throw std::invalid_argument("K must be >= num_objectives");
        }

        // 1. 초기화 (Lines 1-3)
        initialize_();

        // 2. K-n 번 반복 (Lines 4-10)
        for (int k = num_objectives_; k < K; ++k) {
            std::cout << "\n--- Iteration " << (k + 1) << "/" << K << " ---" << std::endl;

            // 3. 최대 후회를 갖는 w* 찾기 (Line 5)
            //    (내부적으로 모든 N에 대해 LP를 풂, Eq. 12)
            MaxRegretResult regret_result = find_max_regret_weight_();
            
            if (!regret_result.found) {
                std::cout << "Could not find a new weight to add. Stopping." << std::endl;
                break;
            }
            
            std::cout << "  New weight w* found: ";
            print_vec(regret_result.w_star);
            std::cout << std::endl;
            std::cout << "  Max upper bound regret: " << std::fixed << std::setprecision(4) 
                      << regret_result.max_regret << std::endl;

            // 4. w*에 대해 스칼라화된 문제 풀기 (Line 6)
            SolutionResult result_star = solve_problem_func_(regret_result.w_star);

            // 5. Omega 와 S_Omega 업데이트 (Lines 7-8)
            Omega_.push_back(regret_result.w_star);
            S_Omega_[regret_result.w_star] = result_star;

            // 6. Neighborhood 분할 (Line 9)
            update_neighborhoods_(regret_result.neighborhood_to_split, regret_result.w_star);
        }
        
        std::cout << "\nMRPS Algorithm finished." << std::endl;
    }

    // --- 결과 접근자 (Getters) ---
    const std::vector<WeightVector>& get_omega() const { return Omega_; }
    const std::map<WeightVector, SolutionResult>& get_s_omega() const { return S_Omega_; }

private:
    /**
     * @brief 알고리즘 1의 초기화 단계 (Lines 1-3)
     */
    void initialize_() {
        std::cout << "Initializing with " << num_objectives_ << " unit vectors..." << std::endl;
        Neighborhood initial_neighborhood_vertices;

        for (int i = 0; i < num_objectives_; ++i) {
            WeightVector w(num_objectives_, 0.0);
            w[i] = 1.0;
            
            SolutionResult result = solve_problem_func_(w);
            Omega_.push_back(w);
            S_Omega_[w] = result;
            initial_neighborhood_vertices.push_back(w);
        }
        
        // 초기 Neighborhood는 W 전체를 나타내는 하나의 심플렉스
        Neighborhoods_.push_back(initial_neighborhood_vertices);
        std::cout << "Initialization complete." << std::endl;
    }

    /**
     * @brief [TODO] 모든 Neighborhood에 대해 LP(Eq. 12)를 풀어 최대 후회를 찾는 함수
     * * 이 함수는 이 알고리즘의 *핵심*이며, 논문을 참조하여
     * 외부 LP 솔버 라이브러리를 사용해 구현해야 합니다.
     *
     * @return MaxRegretResult (w*, max_regret, N*)
     */
    MaxRegretResult find_max_regret_weight_() {
        std::cout << "  [Stub] Finding max regret weight (requires LP solver)..." << std::endl;
        
        // --- 자리표시자 (Placeholder) 로직 ---
        // 실제 구현:
        // 1. best_result.max_regret = -infinity
        // 2. for (const auto& N : Neighborhoods_) {
        // 3.    current_w, current_regret = solve_lp_for_neighborhood(N, S_Omega_); // (Eq. 12 구현)
        // 4.    if (current_regret > best_result.max_regret) {
        // 5.       best_result = {current_w, current_regret, N, true};
        // 6.    }
        // 7. }
        // 8. return best_result;
        
        if (Neighborhoods_.empty()) {
            return MaxRegretResult{ .found = false };
        }

        // *아래는 실제 로직이 아닌, 컴파일을 위한 임시 스텁(stub)입니다.*
        // 첫 번째 Neighborhood를 선택하고, 그 꼭짓점들의 평균을 w*로 반환합니다.
        Neighborhood neighborhood_to_split = Neighborhoods_[0];
        WeightVector w_star(num_objectives_, 0.0);

        for (const auto& vertex : neighborhood_to_split) {
            for (int i = 0; i < num_objectives_; ++i) {
                w_star[i] += vertex[i];
            }
        }
        
        double sum = std::accumulate(w_star.begin(), w_star.end(), 0.0);
        if (sum > 1e-9) {
            for (int i = 0; i < num_objectives_; ++i) {
                w_star[i] /= sum;
            }
        }
        
        // 임시 반환 값 (후회 값 0.1)
        return MaxRegretResult{ w_star, 0.1, neighborhood_to_split, true };
    }

    /**
     * @brief [TODO] w*를 추가하여 기존 Neighborhood를 n개의 더 작은 Neighborhoods로 분할
     * * 이 작업은 가중치 공간(W)의 심플렉스 분할(simplicial partitioning)을
     * 관리하는 것을 의미합니다.
     */
    void update_neighborhoods_(const Neighborhood& neighborhood_to_split, 
                               const WeightVector& new_weight) {
        std::cout << "  [Stub] Updating neighborhoods (requires simplex partitioning)..." << std::endl;
        
        // --- 자리표시자 (Placeholder) 로직 ---
        // 1. `Neighborhoods_` 리스트에서 `neighborhood_to_split`을 찾아서 제거합니다.
        //    (std::find_if 와 std::vector::erase 사용)
        
        // 2. `neighborhood_to_split`의 꼭짓점(n개)을 `new_weight`로 하나씩
        //    교체하여 `n`개의 새로운 Neighborhood를 만듭니다.
        
        // 3. 이 `n`개의 새 Neighborhood를 `Neighborhoods_` 리스트에 추가합니다.
        
        // 예시 로직 (매우 단순화됨):
        // auto it = std::find_if(Neighborhoods_.begin(), Neighborhoods_.end(), 
        //     [&](const Neighborhood& n){ return n == neighborhood_to_split; });
        // 
        // if (it != Neighborhoods_.end()) {
        //     Neighborhoods_.erase(it);
        // 
        //     for (int i = 0; i < num_objectives_; ++i) {
        //         Neighborhood new_n;
        //         for (int j = 0; j < num_objectives_; ++j) {
        //             if (i == j) {
        //                 new_n.push_back(new_weight);
        //             } else {
        //                 new_n.push_back(neighborhood_to_split[j]);
        //             }
        //         }
        //         Neighborhoods_.push_back(new_n);
        //     }
        // }
    }

    // --- 멤버 변수 ---
    int num_objectives_;                 // 목적 함수 개수 (n)
    SolverFunction solve_problem_func_;  // 사용자 제공 솔버 함수
    
    std::vector<WeightVector> Omega_;    // 샘플링된 가중치 벡터(w) 집합
    
    // key: w, value: (솔루션, 목적 함수 값 벡터 f(s*))
    std::map<WeightVector, SolutionResult> S_Omega_;
    
    // 가중치 공간(W)을 분할하는 심플렉스(Neighborhood) 집합
    std::vector<Neighborhood> Neighborhoods_; 
};


// --- 알고리즘 실행 예시 ---

/**
 * @brief [사용자가 직접 구현해야 하는 함수]
 * * 가중치 벡터(w)를 받아, 로봇 계획 문제를 풉니다.
 * 예: w[0]*f1(s) + w[1]*f2(s) + ... 를 최소화하는 솔루션(s)과
 * 그 솔루션의 목적 함수 값 *벡터* [f1(s), f2(s), ...]를 반환합니다.
 */
SolutionResult my_robot_planner(const WeightVector& weight_vector) {
    std::cout << "    Solving for w=";
    print_vec(weight_vector);
    std::cout << "..." << std::endl;
    
    // --- 가상(dummy) 솔루션 ---
    // 실제로는 여기서 RRT*, A* 등의 플래너를 호출해야 합니다.
    SolutionResult result;
    result.solution_data = std::string("dummy_path_data_for_w"); // 솔루션은 문자열이라 가정

    // 2-objective 예시 (Python 버전과 동일한 로직)
    if (weight_vector.size() == 2) {
        double f1 = (1.0 - weight_vector[0]) * 100 + (weight_vector[0]) * 10;
        double f2 = (1.0 - weight_vector[1]) * 50 + (weight_vector[1]) * 20;
        result.objective_vector = {f1, f2};
    } else {
        // n-objective 일반화 (단순 예시)
        result.objective_vector.resize(weight_vector.size());
        for(size_t i = 0; i < weight_vector.size(); ++i) {
            result.objective_vector[i] = (1.0 - weight_vector[i]) * 50 + weight_vector[i] * 10;
        }
    }
    
    return result;
}


int main() {
    const int NUM_OBJECTIVES = 2;  // 예: (거리, 위험도) 2개의 목적 함수
    const int K_SAMPLES = 5;       // 총 5개의 샘플을 뽑음

    try {
        // 1. MRPS 객체 생성 (사용자 정의 솔버 함수를 전달)
        MRPS mrps_sampler(NUM_OBJECTIVES, my_robot_planner);
        
        // 2. 알고리즘 실행
        mrps_sampler.run(K_SAMPLES);

        // 3. 결과 출력
        std::cout << "\n--- Final Results ---" << std::endl;
        std::cout << "Omega (Sampled Weights):" << std::endl;
        for (const auto& w : mrps_sampler.get_omega()) {
            std::cout << "  ";
            print_vec(w);
            std::cout << std::endl;
        }

        std::cout << "\nS_Omega (Solutions & Objective Vectors):" << std::endl;
        for (const auto& pair : mrps_sampler.get_s_omega()) {
            std::cout << "  w=";
            print_vec(pair.first); // key (w)
            std::cout << " -> f(s*)=";
            print_vec(pair.second.objective_vector); // value.objective_vector
            std::cout << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
