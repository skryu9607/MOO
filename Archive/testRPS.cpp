// ==========================================
        // 1. PRE-CHECK FOR DUPLICATES (Save CPU Time)
        // ==========================================
        struct PlanResult {
            Vector f;
            Vector w;
        };
        std::vector<PlanResult> batch_results(batch_to_process.size());
        
        // This array keeps track of which neighborhoods are valid (true) or degenerate (false)
        std::vector<bool> is_valid_task(batch_to_process.size(), true);
        double threshold_duplicate = 0.001;

        for (int i = 0; i < batch_to_process.size(); ++i) {
            for (const auto& entry : database) {
                double dist_weight = std::sqrt(
                    std::pow(entry.w[0] - batch_to_process[i].new_w[0], 2) + 
                    std::pow(entry.w[1] - batch_to_process[i].new_w[1], 2) + 
                    std::pow(entry.w[2] - batch_to_process[i].new_w[2], 2)
                );
                
                if (dist_weight < threshold_duplicate) {
                    is_valid_task[i] = false; // Mark as duplicate!
                    break;
                }
            }
        }

        // ==========================================
        // 2. PARALLEL OMPL PLANNING
        // ==========================================
        #pragma omp parallel for schedule(dynamic)
        for(int i = 0; i < batch_to_process.size(); ++i) {
            int tid = omp_get_thread_num();
            
            // IF DUPLICATE: Skip OMPL entirely to save massive CPU time
            if (!is_valid_task[i]) {
                #pragma omp critical
                {
                    std::cout << "   [Thread " << tid << "] Skipped OMPL for degenerate sample " << i+1 << std::endl;
                }
                continue; 
            }

            // Normal Planning for valid tasks
            Vector f_res = solveBatchItem(batch_to_process[i].new_w, tid, global_max_costs);
            batch_results[i] = {f_res, batch_to_process[i].new_w};

            #pragma omp critical
            {
                std::cout << "   [Thread " << tid << "] Finished sample " << i+1 << "/" << batch_to_process.size() << std::endl;
            }
        }

        // ==========================================
        // 3. SEQUENTIAL UPDATE & TRIANGLE SPLITTING
        // ==========================================
        for (int i = 0; i < batch_to_process.size(); ++i) {
            // THE BOLD DECISION: Discard the degenerate neighborhood
            if (!is_valid_task[i]) {
                std::cout << "Discarding degenerate neighborhood " << i+1 << " (Duplicate weight found)." << std::endl;
                // Using continue here skips adding it to the database 
                // AND skips creating the 3 zero-area sub-triangles!
                continue; 
            }

            const auto* task = &batch_to_process[i];
            const auto& res  = batch_results[i];
            
            int new_id = (int)database.size();
            database.push_back({new_id, res.w, res.f});
            
            int d = task->id_d;
            int r = task->id_r;
            int t = task->id_t;
            
            std::vector<double> w_d = database[d].w;
            std::vector<double> w_r = database[r].w;
            std::vector<double> w_t = database[t].w;

            // Log
            logFile << k << "," << res.w[0] << "," << res.w[1] << "," << res.w[2] << ", "
                    << res.f[0] << "," << res.f[1] << "," << res.f[2] << ", " 
                    << task->max_regret << "," << w_d[0] << "," << w_d[1] << "," 
                    << w_d[2] << "," << w_r[0] << "," << w_r[1] << "," << w_r[2] 
                    << "," << w_t[0] << "," << w_t[1] << "," << w_t[2] << "\n";

            // Define the 3 new triangles 
            int sets[3][3] = {
                {d, r, new_id},
                {d, new_id, t},
                {new_id, r, t}
            };

            for (int j = 0; j < 3 ; ++j) {
                Neighborhood n_child;
                n_child.id_d = sets[j][0];
                n_child.id_r = sets[j][1];
                n_child.id_t = sets[j][2];

                std::vector<SampledCost> corners = {
                    database[n_child.id_d],
                    database[n_child.id_r],
                    database[n_child.id_t]
                };

                // Solve LP for new sub-neighborhood
                RegretResult lp_res = solveMaxRegretLP(corners, global_max_costs);
                
                n_child.new_w = lp_res.worst_w;
                n_child.max_regret = lp_res.max_regret;

                // Only add if it has meaningful regret
                if (n_child.max_regret > 1e-6) {
                    neighborhoods.push_back(n_child);
                }
            }
        }
