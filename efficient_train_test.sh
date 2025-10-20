#!/bin/bash
# 高效训练和测试流程示例

# ============================================
# 方案1: 边训练边测试（推荐用于快速验证）
# ============================================

# 训练域6并在后台运行
screen -dmS unet-d6 bash -c "
    python train_unet_only.py \
        --dataset prostate \
        --lb_domain 6 \
        --lb_num 20 \
        --save_name UNet_only_d6_8000 \
        --gpu 2 \
        --save_model \
        --overwrite \
        --max_iterations 8000 && \
    python test_unet_only.py \
        --dataset prostate \
        --save_name UNet_only_d6_8000 \
        --test_domains 6 \
        --gpu 2
"

# ============================================
# 方案2: 批量训练，逐个快速测试
# ============================================

# 启动所有域的训练
for domain in {1..6}; do
    gpu=$((domain % 4))  # 分配GPU：0,1,2,3循环
    screen -dmS unet-d${domain} bash -c "
        python train_unet_only.py \
            --dataset prostate \
            --lb_domain ${domain} \
            --lb_num 20 \
            --save_name UNet_only_d${domain}_8000 \
            --gpu ${gpu} \
            --save_model \
            --overwrite \
            --max_iterations 8000
    "
    echo "Started training for domain ${domain} on GPU ${gpu}"
    sleep 5  # 避免同时启动太多进程
done

# 等待一段时间后，检查并测试已完成的域
echo "Waiting for training to complete..."
sleep 3600  # 等待1小时（根据实际训练时间调整）

# 逐个检查并测试
for domain in {1..6}; do
    model_path="../model/prostate/train_unet_only/UNet_only_d${domain}_8000/unet_avg_dice_best_model.pth"
    if [ -f "$model_path" ]; then
        echo "Testing domain ${domain}..."
        python test_unet_only.py \
            --dataset prostate \
            --save_name UNet_only_d${domain}_8000 \
            --test_domains ${domain} \
            --gpu 0
    else
        echo "Model for domain ${domain} not ready yet"
    fi
done

# ============================================
# 方案3: 训练完成后的完整测试
# ============================================

# 假设所有域都已训练完成，现在进行完整测试

# 测试每个域在所有域上的表现（跨域泛化能力分析）
for train_domain in {1..6}; do
    echo "=========================================="
    echo "Testing model trained on domain ${train_domain}"
    echo "=========================================="
    
    # 测试在训练域上的表现
    echo "On training domain ${train_domain}:"
    python test_unet_only.py \
        --dataset prostate \
        --save_name UNet_only_d${train_domain}_8000 \
        --test_domains ${train_domain} \
        --gpu 0
    
    # 测试在其他域上的泛化能力
    echo "On other domains:"
    python test_unet_only.py \
        --dataset prostate \
        --save_name UNet_only_d${train_domain}_8000 \
        --test_domains all \
        --gpu 0
    
    sleep 10
done

# ============================================
# 方案4: 针对性测试（消融实验）
# ============================================

# 测试近源域vs远源域
echo "Testing near-domain generalization..."
# 域6训练，测试5,6（相近域）
python test_unet_only.py \
    --dataset prostate \
    --save_name UNet_only_d6_8000 \
    --test_domains 5,6 \
    --gpu 0

echo "Testing far-domain generalization..."
# 域6训练，测试1,2（远域）
python test_unet_only.py \
    --dataset prostate \
    --save_name UNet_only_d6_8000 \
    --test_domains 1,2 \
    --gpu 0

# ============================================
# 方案5: 快速验证单个实验
# ============================================

# 训练一个域，快速验证效果
echo "Quick validation experiment..."
python train_unet_only.py \
    --dataset prostate \
    --lb_domain 1 \
    --lb_num 20 \
    --save_name UNet_quick_test \
    --gpu 0 \
    --save_model \
    --overwrite \
    --max_iterations 1000

# 立即测试训练域
python test_unet_only.py \
    --dataset prostate \
    --save_name UNet_quick_test \
    --test_domains 1 \
    --gpu 0

# ============================================
# 方案6: 对比不同训练策略
# ============================================

# 训练两个不同配置的模型
echo "Training model A (8000 iterations)..."
python train_unet_only.py \
    --dataset prostate \
    --lb_domain 6 \
    --lb_num 20 \
    --save_name UNet_d6_8k \
    --gpu 0 \
    --save_model \
    --overwrite \
    --max_iterations 8000

echo "Training model B (16000 iterations)..."
python train_unet_only.py \
    --dataset prostate \
    --lb_domain 6 \
    --lb_num 20 \
    --save_name UNet_d6_16k \
    --gpu 0 \
    --save_model \
    --overwrite \
    --max_iterations 16000

# 对比两个模型在同一测试集上的表现
echo "Comparing models on test domain 6..."
echo "Model A (8k iterations):"
python test_unet_only.py \
    --dataset prostate \
    --save_name UNet_d6_8k \
    --test_domains 6 \
    --gpu 0

echo "Model B (16k iterations):"
python test_unet_only.py \
    --dataset prostate \
    --save_name UNet_d6_16k \
    --test_domains 6 \
    --gpu 0

# ============================================
# 方案7: 并行测试多个模型
# ============================================

# 假设有4个GPU，并行测试不同模型
for domain in {1..4}; do
    gpu=$((domain - 1))
    python test_unet_only.py \
        --dataset prostate \
        --save_name UNet_only_d${domain}_8000 \
        --test_domains ${domain} \
        --gpu ${gpu} &
done
wait  # 等待所有测试完成
echo "All parallel tests completed!"

# ============================================
# 方案8: 增量测试（节省时间）
# ============================================

# 第一阶段：只测试训练域（快速验证）
echo "Phase 1: Testing training domains only..."
for domain in {1..6}; do
    python test_unet_only.py \
        --dataset prostate \
        --save_name UNet_only_d${domain}_8000 \
        --test_domains ${domain} \
        --gpu 0
done

# 第二阶段：只对表现好的模型进行完整测试
echo "Phase 2: Full testing for best models..."
# 假设域3和域6表现最好
for domain in 3 6; do
    python test_unet_only.py \
        --dataset prostate \
        --save_name UNet_only_d${domain}_8000 \
        --test_domains all \
        --gpu 0
done

# ============================================
# 实用函数
# ============================================

# 函数：检查模型是否训练完成
check_model_ready() {
    local dataset=$1
    local save_name=$2
    local model_path="../model/${dataset}/train_unet_only/${save_name}/unet_avg_dice_best_model.pth"
    
    if [ -f "$model_path" ]; then
        echo "Model $save_name is ready"
        return 0
    else
        echo "Model $save_name is not ready"
        return 1
    fi
}

# 函数：自动测试准备好的模型
auto_test_when_ready() {
    local dataset=$1
    local save_name=$2
    local test_domains=$3
    local gpu=$4
    
    while ! check_model_ready "$dataset" "$save_name"; do
        echo "Waiting for model to be ready..."
        sleep 300  # 每5分钟检查一次
    done
    
    echo "Model ready! Starting test..."
    python test_unet_only.py \
        --dataset "$dataset" \
        --save_name "$save_name" \
        --test_domains "$test_domains" \
        --gpu "$gpu"
}

# 使用示例
# auto_test_when_ready prostate UNet_only_d6_8000 6 0 &
