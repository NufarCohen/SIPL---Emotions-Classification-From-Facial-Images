import os
import time

print("==============================================================")
print("     STARTING FULL STACKING ENSEMBLE PIPELINE     ")
print("==============================================================\n")

total_start_time = time.time()

print("========== 1/3: STARTING - Training Base Models ==========")
start_step1 = time.time()
os.system("python train_base_models_heterogeneous.py") 
end_step1 = time.time()
print(f"========== 1/3: COMPLETE - Base Models Trained (Time: {(end_step1 - start_step1)/3600:.2f} hours) ==========\n")


print("========== 2/3: STARTING - Creating Feature Dataset ==========")
start_step2 = time.time()
os.system("python create_features_hetro.py")
end_step2 = time.time()
print(f"========== 2/3: COMPLETE - Feature Dataset Created (Time: {(end_step2 - start_step2)/60:.1f} minutes) ==========\n")


print("========== 3/3: STARTING - Training Meta-Learner ==========")
start_step3 = time.time()
os.system("python train_meta_hetero.py")
end_step3 = time.time()
print(f"========== 3/3: COMPLETE - Meta-Learner Trained (Time: {(end_step3 - start_step3)/60:.1f} minutes) ==========\n")


total_end_time = time.time()
print("==============================================================")
print("     PIPELINE FINISHED SUCCESSFULLY!     ")
print(f"     Total Runtime: {(total_end_time - total_start_time) / 3600:.2f} hours     ")
print("==============================================================")