import matplotlib.pyplot as plt

def plot_matching_graph(ground_truths, predictions):
    # 시각화 (첫 번째 샘플)
    plt.figure(figsize=(10, 6))
    plt.plot(ground_truths[:100], label='Ground Truth')
    plt.plot(predictions[:100], label='Predictions')
    plt.title("Model Predictions vs Ground Truth")
    plt.xlabel("Sample Index")
    plt.ylabel("Values")
    plt.legend()
    plt.grid()
    plt.show()

def plot_boxplots(ground_truths, predictions):
    # 구간 설정
    bins = [4.0, 5.0, 6.0, 7.0, 8.0]  # 구간 경계 (mEq/L)
    labels = ["<4.0", "4.0-5.0", "5.0-6.0", "6.0-7.0", "7.0-8.0", ">8.0"]

    # 구간별 데이터 분류
    binned_predictions = [[] for _ in range(len(labels))]
    for truth, pred in zip(ground_truths, predictions):
        for i, (lower, upper) in enumerate(zip([float("-inf")] + bins, bins + [float("inf")])):
            if lower <= truth < upper:
                binned_predictions[i].append(pred)
                break

    # 박스 플롯 그리기
    plt.figure(figsize=(10, 6))
    plt.boxplot(binned_predictions, labels=labels, patch_artist=True)
    plt.title("Box Plot of Predictions by Ground Truth Ranges")
    plt.xlabel("Potassium Concentration Ranges (mEq/L)")
    plt.ylabel("Predicted Values")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def print_dataset_info(ecg,spl,pat):
    samples_1 = [(i, e, s, p) for i, (e, s, p) in enumerate(zip(ecg, spl, pat)) if s < 4.0]
    samples_2 = [(i, e, s, p) for i, (e, s, p) in enumerate(zip(ecg, spl, pat)) if 4.0 <= s < 5.0]
    samples_3 = [(i, e, s, p) for i, (e, s, p) in enumerate(zip(ecg, spl, pat)) if 5.0 <= s < 6.0]
    samples_4 = [(i, e, s, p) for i, (e, s, p) in enumerate(zip(ecg, spl, pat)) if 6.0 <= s < 7.0]
    samples_5 = [(i, e, s, p) for i, (e, s, p) in enumerate(zip(ecg, spl, pat)) if 7.0 <= s < 8.0]
    samples_6 = [(i, e, s, p) for i, (e, s, p) in enumerate(zip(ecg, spl, pat)) if s >= 8.0]
    print(f" < 4.0 mEq/L dataset: ", len(samples_1))
    print(f" 4.0 mEq/L - 5.0 mEq/L dataset: ", len(samples_2))
    print(f" 5.0 mEq/L - 6.0 mEq/L dataset: ", len(samples_3))
    print(f" 6.0 mEq/L - 7.0 mEq/L dataset: ", len(samples_4))
    print(f" 7.0 mEq/L - 8.0 mEq/L dataset: ", len(samples_5))
    print(f" > 8.0 mEq/L dataset: ", len(samples_6))

def plot_heatmaps(ground_truths, predictions):
    # 2. 2D 히스토그램 데이터 계산 (밀도 히트맵)
    # 2. 스캐터 플롯
    plt.figure(figsize=(8, 8))

    # 밀도 배경 추가 (선택 사항, 더 나은 시각화)
    plt.hist2d(
        ground_truths,
        predictions,
        bins=30,
        cmap="Blues"
    )
    plt.colorbar(label='Density')
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.title("Density Heatmap")
    plt.show()

    # 스캐터 플롯
    plt.scatter(ground_truths, predictions, color="blue", alpha=0.6, s=10, label="Data Points")

    # 대각선 추가 (완벽히 일치하는 값)
    plt.plot([4, 9], [4, 9], 'r-', linewidth=2, label="Perfect Match")

    # 3. 그래프 스타일링
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.title("Scatter Plot with Density Background")
    plt.xticks(range(4, 10))
    plt.yticks(range(4, 10))
    plt.grid(color='green', linestyle='--', linewidth=1, alpha=0.5)
    plt.legend()
    plt.show()