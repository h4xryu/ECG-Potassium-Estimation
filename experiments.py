from utils import visualize, plot_boxplots, print_dataset_info, plot_heatmaps, one_cycle, signal_collector, plot_matching_graph
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from train import *


@Experiment
def exp1(root='./dataset', epoch=100):
    '''
    실험 1. 가장 작은 시퀀스 길이로 자르기
    '''

    ecg_a, pat_a, spl_a = get_ecg_from_lead(root=root, cls='A', lead='v5')
    ecg_b, pat_b, spl_b = get_ecg_from_lead(root=root, cls='B', lead='v5')
    ecg = ecg_a + ecg_b
    pat = pat_a + pat_b
    spl = spl_a + spl_b

    for idx in range(len(spl)):
        print("Serum Potassium Level: ", spl[idx])
        print("Patient No.: ", pat[idx])

        plot_signal_from_data(ecg[idx])


@Experiment
def exp2(root='./dataset', epoch=100):
    '''
    실험 2. PQRST Detection한 모델
    '''

    # generate_datas(root=root)

    # seqlen = 150
    ecg_a, pat_a, spl_a = get_ecg_from_lead(root=root, cls='A', lead='v5')
    ecg_b, pat_b, spl_b = get_ecg_from_lead(root=root, cls='B', lead='v5')
    ecg= ecg_a + ecg_b
    pat= pat_a + pat_b
    spl = spl_a + spl_b


    ecg_i, pat_i, spl_i = get_ecg_from_lead(root=root, cls='A', lead='v5')
    ecg_j, pat_j, spl_j = get_ecg_from_lead(root=root, cls='B', lead='v5')
    potassium_levels_5 = spl_i + spl_j
    bins = np.linspace(min(potassium_levels_5), max(potassium_levels_5), 20)  # 10개의 구간으로 나눔

    # 히스토그램 생성
    plt.hist(potassium_levels_5, bins=bins, edgecolor='black')
    plt.title('Distribution of Serum Potassium Levels (V5)')
    plt.xlabel('(mmol/L)')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # ecg = ecg_a + ecg_b + ecg_c + ecg_d + ecg_e + ecg_f + ecg_g + ecg_h + ecg_i + ecg_j + ecg_k + ecg_l
    # pat = pat_a + pat_b + pat_c + pat_d + pat_e + pat_f + pat_g + pat_h + pat_i + pat_j + pat_k + pat_l
    # spl = spl_a + spl_b + spl_c + spl_d + spl_e + spl_f + spl_g + spl_h + spl_i + spl_j + spl_k + spl_l

    fs = 150  # 샘플링 주파수 (Hz)

    data = np.load(os.getcwd() + '/temp_selected_data_2.npz', allow_pickle=True)
    print(data.files)
    ecg = data['signals']
    spl = data['potassium_levels']
    spl_tmp = spl
    # spl = [encode_k_concentration(k) for i, k in enumerate(spl)]

    pat = data['patient_ids']

    # 훈련된 모델, 손실 함수, 테스트 로더 필요
    criterion = nn.L1Loss()

    # 모델 테스트 실행

    model = train(ecg, spl, criterion, num_epochs=epoch)
    test_loss, predictions, ground_truths = test_model(ecg,spl,model, criterion)


    # 시각화 (첫 번째 샘플)
    plt.figure(figsize=(10, 6))
    plt.plot(ground_truths[:len(predictions)], label='Ground Truth')
    plt.plot(predictions[:len(predictions)], label='Predictions')
    plt.title("Model Predictions vs Ground Truth")
    plt.xlabel("Sample Index")
    plt.ylabel("Values")
    plt.legend()
    plt.grid()
    plt.show()

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(ground_truths, predictions)
    mse = mean_squared_error(ground_truths, predictions)
    r2 = r2_score(ground_truths, predictions)
    print("dataset x",len(spl))
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R²): {r2:.4f}")




@Experiment
def exp3(root='./dataset', epoch=100):
    # generate_SPL_datas(root)
    # generate_datas(root=root)
    seqlen = 150
    ecg_a, pat_a, spl_a = get_ecg_from_lead(root=root, cls='A', lead='v1')
    ecg_b, pat_b, spl_b = get_ecg_from_lead(root=root, cls='B', lead='v1')
    ecg_c, pat_c, spl_c = get_ecg_from_lead(root=root, cls='A', lead='v2')
    ecg_d, pat_d, spl_d = get_ecg_from_lead(root=root, cls='B', lead='v2')
    ecg_e, pat_e, spl_e = get_ecg_from_lead(root=root, cls='A', lead='v3')
    ecg_f, pat_f, spl_f = get_ecg_from_lead(root=root, cls='B', lead='v3')
    ecg_g, pat_g, spl_g = get_ecg_from_lead(root=root, cls='A', lead='v4')
    ecg_h, pat_h, spl_h = get_ecg_from_lead(root=root, cls='B', lead='v4')
    ecg_i, pat_i, spl_i = get_ecg_from_lead(root=root, cls='A', lead='v5')
    ecg_j, pat_j, spl_j = get_ecg_from_lead(root=root, cls='B', lead='v5')
    ecg_k, pat_k, spl_k = get_ecg_from_lead(root=root, cls='A', lead='v6')
    ecg_l, pat_l, spl_l = get_ecg_from_lead(root=root, cls='B', lead='v6')

    ecg = ecg_a + ecg_b + ecg_c + ecg_d + ecg_e + ecg_f + ecg_g + ecg_h + ecg_i + ecg_j + ecg_k + ecg_l
    pat = pat_a + pat_b + pat_c + pat_d + pat_e + pat_f + pat_g + pat_h + pat_i + pat_j + pat_k + pat_l
    spl = spl_a + spl_b + spl_c + spl_d + spl_e + spl_f + spl_g + spl_h + spl_i + spl_j + spl_k + spl_l




    model = train(ecg, spl,num_epochs=epoch)

    # 훈련된 모델, 손실 함수, 테스트 로더 필요
    criterion = nn.L1Loss()

    # 모델 테스트 실행

    test_loss, predictions, ground_truths = test_model(ecg, spl, model, criterion)

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

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(ground_truths, predictions)
    mse = mean_squared_error(ground_truths, predictions)
    r2 = r2_score(ground_truths, predictions)

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R²): {r2:.4f}")

    # 가정: ground_truths와 predictions는 테스트 함수 실행 결과에서 가져옴
    ground_truths = np.array(ground_truths).flatten()
    predictions = np.array(predictions).flatten()


@Experiment
def exp4(root='./dataset', epoch=100):
    # generate_datas(root=root)

    ecg_a, pat_a, spl_a = get_ecg_from_lead(root=root, cls='A', lead='v1')
    ecg_b, pat_b, spl_b = get_ecg_from_lead(root=root, cls='B', lead='v1')

    ecg = ecg_a + ecg_b
    pat = pat_a + pat_b
    spl = spl_a + spl_b

    ecg, spl, pat = one_cycle(ecg,spl,pat)

    # criterion = nn.HuberLoss()
    criterion = nn.HuberLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCRNNModel().to(device)
    model = train(model, device, ecg, spl, criterion, num_epochs=epoch)
    # 모델 테스트 실행

    test_loss, predictions, ground_truths = test_model(ecg, spl, model, criterion)


    mae = mean_absolute_error(ground_truths, predictions)
    mse = mean_squared_error(ground_truths, predictions)
    r2 = r2_score(ground_truths, predictions)

    print_dataset_info(ecg, spl, pat)

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R²): {r2:.4f}")


    # 가정: ground_truths와 predictions는 테스트 함수 실행 결과에서 가져옴
    ground_truths = np.array(ground_truths).flatten()
    predictions = np.array(predictions).flatten()
    print_dataset_info(ecg, spl, pat)
    plot_boxplots(ground_truths, predictions)
    plot_heatmaps(ground_truths, predictions)



@Experiment
def exp5(root='./dataset', epoch=100): #long sequence
    # generate_datas(root=root)

    ecg_a, pat_a, spl_a = get_ecg_from_lead(root=root, cls='A', lead='v1')
    ecg_b, pat_b, spl_b = get_ecg_from_lead(root=root, cls='B', lead='v1')

    ecg = ecg_a + ecg_b
    spl = spl_a + spl_b
    pat = pat_a + pat_b


    ecg,spl, pat = one_cycle(ecg, spl, pat)


    # criterion = nn.HuberLoss()
    criterion = nn.L1Loss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCRNNModel_longseq().to(device)
    model = train(model, device, ecg, spl, criterion, num_epochs=epoch)
    # 모델 테스트 실행

    test_loss, predictions, ground_truths = test_model(ecg, spl, model, criterion)




    mae = mean_absolute_error(ground_truths, predictions)
    mse = mean_squared_error(ground_truths, predictions)
    r2 = r2_score(ground_truths, predictions)



    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R²): {r2:.4f}")


    # 가정: ground_truths와 predictions는 테스트 함수 실행 결과에서 가져옴
    ground_truths = np.array(ground_truths).flatten()
    predictions = np.array(predictions).flatten()

    print_dataset_info(ecg, spl, pat)
    plot_boxplots(ground_truths, predictions)
    plot_heatmaps(ground_truths, predictions)






