clc;
clear;
close all;
rng(42); % Reproductibilitate

%% =========================
%  PARAMETRI SIMULARE
% ==========================
numCells = 3;
numDevices = 60;
devicesPerCell = numDevices / numCells;

numTimeSteps = 2000;
dt = 1; % 1 secunda per pas

carrierFreq_GHz = 3.5;
bandwidth_Hz = 20e6;           % 20 MHz / celula
txPower_dBm = 33;              % putere gNB
noiseFigure_dB = 5;
thermalNoise_dBm = -174 + 10*log10(bandwidth_Hz) + noiseFigure_dB;

cellRadius = 250;              % metri
alphaEfficiency = 0.7;         % eficienta spectrala practica
pathLossExp = 3.2;             % exponent path loss
shadowStdDev = 4;              % shadowing in dB

%% =========================
%  POZITII gNB-uri
% ==========================
gNB_positions = [
    0,   0;
    500, 0;
    250, 433
];

%% =========================
%  GENERARE DISPOZITIVE IoT
% ==========================
devicePos = zeros(numDevices, 2);
deviceHomeCell = zeros(numDevices, 1);
trafficClass = zeros(numDevices, 1); % 1=periodic, 2=event-driven, 3=high-load

idx = 1;
for c = 1:numCells
    centerX = gNB_positions(c,1);
    centerY = gNB_positions(c,2);

    for d = 1:devicesPerCell
        r = cellRadius * sqrt(rand()); % distributie uniforma in disc
        theta = 2*pi*rand();

        x = centerX + r*cos(theta);
        y = centerY + r*sin(theta);

        devicePos(idx,:) = [x, y];
        deviceHomeCell(idx) = c;

        % Atribuire clasa trafic
        p = rand();
        if p < 0.5
            trafficClass(idx) = 1; % periodic
        elseif p < 0.8
            trafficClass(idx) = 2; % event-driven
        else
            trafficClass(idx) = 3; % high-load
        end

        idx = idx + 1;
    end
end

%% =========================
%  PREALOCARI
% ==========================
maxRows = numDevices * numTimeSteps;
results = cell(maxRows, 21);

rowIdx = 1;

%% =========================
%  SIMULARE PRINCIPALA
% ==========================
for t = 1:numTimeSteps

    % Activitate dispozitive in acest pas
    isActive = false(numDevices,1);
    packetSize_bytes = zeros(numDevices,1);
    packetRate = zeros(numDevices,1);

    for i = 1:numDevices
        switch trafficClass(i)
            case 1 % periodic
                if mod(t, randi([5,10])) == 0
                    isActive(i) = true;
                end
                packetSize_bytes(i) = randi([100, 200]);
                packetRate(i) = 1 / randi([5,10]);

            case 2 % event-driven
                if rand() < 0.15
                    isActive(i) = true;
                end
                packetSize_bytes(i) = randi([200, 500]);
                packetRate(i) = rand() * 0.5;

            case 3 % high-load
                if rand() < 0.5
                    isActive(i) = true;
                end
                packetSize_bytes(i) = randi([1000, 5000]);
                packetRate(i) = rand() * 2 + 0.5;
        end
    end

    % Distante si puteri receptionate pentru toate dispozitivele
    rxPower_dBm_all = zeros(numDevices, numCells);
    distances_all = zeros(numDevices, numCells);

    for i = 1:numDevices
        for c = 1:numCells
            dx = devicePos(i,1) - gNB_positions(c,1);
            dy = devicePos(i,2) - gNB_positions(c,2);
            d = sqrt(dx^2 + dy^2);

            % Evitare log10(0)
            d = max(d, 1);

            distances_all(i,c) = d;

            % Path loss simplificat
            PL_dB = 32.4 + 20*log10(carrierFreq_GHz*1000) + 10*pathLossExp*log10(d/1000);
            shadowing_dB = shadowStdDev * randn();

            rxPower_dBm_all(i,c) = txPower_dBm - PL_dB + shadowing_dB;
        end
    end

    % Asociere serving cell pe baza puterii maxime
    [servingRSRP_dBm, servingCell] = max(rxPower_dBm_all, [], 2);

    % Numar utilizatori activi per celula
    activeDevicesCell = zeros(numCells,1);
    for c = 1:numCells
        activeDevicesCell(c) = sum(isActive & (servingCell == c));
    end
    activeDevicesTotal = sum(isActive);

    % Calcul KPI pentru fiecare dispozitiv
    for i = 1:numDevices
        currentServingCell = servingCell(i);

        % Putere utila
        signal_dBm = rxPower_dBm_all(i, currentServingCell);
        signal_mW = 10^(signal_dBm/10);

        % Interferenta = suma puterilor de la celelalte doua celule
        interfererCells = setdiff(1:numCells, currentServingCell);
        interf_mW = 0;
        for k = 1:length(interfererCells)
            interf_dBm = rxPower_dBm_all(i, interfererCells(k));
            interf_mW = interf_mW + 10^(interf_dBm/10);
        end

        noise_mW = 10^(thermalNoise_dBm/10);

        % SINR
        sinr_linear = signal_mW / (interf_mW + noise_mW);
        sinr_dB = 10*log10(sinr_linear);

        % Distante utile
        distServing = distances_all(i, currentServingCell);
        neighborCells = setdiff(1:numCells, currentServingCell);
        distNb1 = distances_all(i, neighborCells(1));
        distNb2 = distances_all(i, neighborCells(2));

        % Path loss serving
        pathLossServing_dB = txPower_dBm - signal_dBm;

        % Cell load
        if sum(servingCell == currentServingCell) > 0
            cellLoad = activeDevicesCell(currentServingCell) / sum(servingCell == currentServingCell);
        else
            cellLoad = 0;
        end

        % Throughput doar pentru dispozitive active
        if isActive(i) && activeDevicesCell(currentServingCell) > 0
            allocatedBandwidth_Hz = bandwidth_Hz / activeDevicesCell(currentServingCell);

            throughput_bps = alphaEfficiency * allocatedBandwidth_Hz * log2(1 + sinr_linear);

            % Optional limitare superioara pentru realism
            throughput_bps = min(throughput_bps, 100e6);
        else
            allocatedBandwidth_Hz = 0;
            throughput_bps = 0;
        end

        % Latenta
        procDelay_ms = 3 + 5*rand(); % 3-8 ms
        queueDelay_ms = 2 + 20*cellLoad + 5*rand();

        if throughput_bps > 0
            txDelay_ms = (packetSize_bytes(i)*8 / throughput_bps) * 1000;
        else
            txDelay_ms = 100; % penalizare mare daca nu transmite
        end

        latency_ms = procDelay_ms + queueDelay_ms + txDelay_ms;

        % PER - functie simplificata de SINR + load + packet size
        normPacket = packetSize_bytes(i) / 5000;
        perVal = exp(-0.15 * max(sinr_linear, 1e-6)) + 0.25*cellLoad + 0.1*normPacket;

        % limitare intre 0 si 1
        perVal = min(max(perVal, 0), 1);

        % Salvare rand
        results(rowIdx, :) = {
            t, ...
            i, ...
            trafficClass(i), ...
            devicePos(i,1), ...
            devicePos(i,2), ...
            currentServingCell, ...
            distServing, ...
            distNb1, ...
            distNb2, ...
            isActive(i), ...
            activeDevicesCell(currentServingCell), ...
            activeDevicesTotal, ...
            cellLoad, ...
            packetSize_bytes(i), ...
            packetRate(i), ...
            allocatedBandwidth_Hz, ...a
            servingRSRP_dBm(i), ...
            sinr_dB, ...
            throughput_bps, ...
            latency_ms, ...
            perVal ...
        };

        rowIdx = rowIdx + 1;
    end
end

%% =========================
%  CONVERSIE LA TABEL
% ==========================
results = results(1:rowIdx-1, :);

varNames = {
    'time_step', ...
    'device_id', ...
    'traffic_class', ...
    'x', ...
    'y', ...
    'serving_cell', ...
    'distance_serving_m', ...
    'distance_neighbor1_m', ...
    'distance_neighbor2_m', ...
    'is_active', ...
    'active_devices_cell', ...
    'active_devices_total', ...
    'cell_load', ...
    'packet_size_bytes', ...
    'packet_rate', ...
    'allocated_bandwidth_Hz', ...
    'RSRP_dBm', ...
    'SINR_dB', ...
    'throughput_bps', ...
    'latency_ms', ...
    'PER'
};

resultsTable = cell2table(results, 'VariableNames', varNames);

%% =========================
%  SALVARE CSV
% ==========================
writetable(resultsTable, 'private_5g_iot_dataset.csv');

disp('Simularea s-a terminat.');
disp('Fisier generat: private_5g_iot_dataset.csv');

%% =========================
%  GRAFICE RAPIDE
% ==========================
figure;
gscatter(devicePos(:,1), devicePos(:,2), deviceHomeCell);
hold on;
plot(gNB_positions(:,1), gNB_positions(:,2), 'kp', 'MarkerSize', 14, 'MarkerFaceColor', 'y');
title('Topologia retelei: dispozitive IoT si gNB-uri');
xlabel('X [m]');
ylabel('Y [m]');
grid on;
legend('Cell 1 Devices', 'Cell 2 Devices', 'Cell 3 Devices', 'gNB');

figure;
histogram(resultsTable.SINR_dB, 40);
title('Distributia SINR');
xlabel('SINR [dB]');
ylabel('Numar observatii');
grid on;

figure;
histogram(resultsTable.throughput_bps / 1e6, 40);
title('Distributia Throughput');
xlabel('Throughput [Mbps]');
ylabel('Numar observatii');
grid on;

figure;
histogram(resultsTable.latency_ms, 40);
title('Distributia Latentei');
xlabel('Latenta [ms]');
ylabel('Numar observatii');
grid on;