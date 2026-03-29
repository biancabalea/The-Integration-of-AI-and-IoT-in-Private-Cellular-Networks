clc;
clear;
close all;

rng(42, 'twister');

%% =========================================================
%  PRIVATE 5G + IoT NETWORK SIMULATION
%  Final MATLAB Version for Dissertation Project
%  Title: "Integrarea AI si IoT in retelele celulare private"
%% =========================================================

params = initSimulationParameters();

fprintf('=== START PRIVATE 5G IoT SIMULATION ===\n');
fprintf('Cells: %d\n', params.numCells);
fprintf('Devices: %d\n', params.numDevices);
fprintf('Time steps: %d\n', params.numTimeSteps);

[gNB_positions, devices] = initializeNetwork(params);

resultsTable = runSimulation(params, gNB_positions, devices);

writeOutputs(resultsTable, params);

fprintf('=== SIMULATION COMPLETED SUCCESSFULLY ===\n');
fprintf('Dataset file: %s\n', params.outputCsvFile);
fprintf('Summary file: %s\n', params.outputSummaryFile);

%% =========================================================
%  LOCAL FUNCTIONS
%% =========================================================

function params = initSimulationParameters()
    params.numCells = 3;
    params.numDevices = 60;
    params.devicesPerCell = params.numDevices / params.numCells;

    params.numTimeSteps = 2000;
    params.dt_s = 1.0;

    params.carrierFreq_GHz = 3.5;
    params.bandwidth_Hz = 20e6;              % 20 MHz per cell
    params.txPower_dBm = 33;                 % gNB transmit power
    params.noiseFigure_dB = 5;               % receiver noise figure
    params.cellRadius_m = 250;
    params.pathLossExponent = 3.2;
    params.shadowStdDev_dB = 4.0;
    params.alphaEfficiency = 0.70;           % practical spectral efficiency factor
    params.maxThroughput_bps = 100e6;        % upper throughput cap per device

    % Delay model
    params.procDelayMin_ms = 3;
    params.procDelayMax_ms = 8;
    params.baseQueueDelay_ms = 2;
    params.queueLoadFactor_ms = 20;
    params.queueRandomness_ms = 5;
    params.inactivePenaltyTxDelay_ms = 100;

    % PER model
    params.perSinrFactor = 0.15;
    params.perLoadFactor = 0.25;
    params.perPacketFactor = 0.10;

    % Thermal noise over full cell bandwidth
    params.thermalNoise_dBm = -174 + 10*log10(params.bandwidth_Hz) + params.noiseFigure_dB;

    % 3 gNB positions (triangular deployment)
    params.gNB_positions = [
        0,   0;
        500, 0;
        250, 433
    ];

    % Traffic class probabilities
    params.probClass1 = 0.50; % periodic
    params.probClass2 = 0.30; % event-driven
    params.probClass3 = 0.20; % high-load

    % Export
    params.outputCsvFile = 'private_5g_iot_dataset_final.csv';
    params.outputSummaryFile = 'private_5g_iot_summary.txt';

    % Traffic behavior
    params.class1PeriodMin = 5;
    params.class1PeriodMax = 10;
    params.class1PacketMin = 100;
    params.class1PacketMax = 200;

    params.class2ActivationProb = 0.15;
    params.class2PacketMin = 200;
    params.class2PacketMax = 500;

    params.class3ActivationProb = 0.50;
    params.class3PacketMin = 1000;
    params.class3PacketMax = 5000;

    % Numerical safety
    params.minDistance_m = 1;
    params.minLinearPower_mW = 1e-12;

    % Metadata
    params.simulationName = 'Private5G_IoT_3Cells_Final';
    params.version = '1.0';
end

function [gNB_positions, devices] = initializeNetwork(params)
    gNB_positions = params.gNB_positions;

    devices.id = (1:params.numDevices).';
    devices.homeCell = zeros(params.numDevices,1);
    devices.x = zeros(params.numDevices,1);
    devices.y = zeros(params.numDevices,1);
    devices.trafficClass = zeros(params.numDevices,1);
    devices.periodicInterval = zeros(params.numDevices,1);

    idx = 1;

    for c = 1:params.numCells
        centerX = gNB_positions(c,1);
        centerY = gNB_positions(c,2);

        for d = 1:params.devicesPerCell
            [x, y] = generateRandomPointInCell(centerX, centerY, params.cellRadius_m);

            devices.x(idx) = x;
            devices.y(idx) = y;
            devices.homeCell(idx) = c;

            p = rand();
            if p < params.probClass1
                devices.trafficClass(idx) = 1;
                devices.periodicInterval(idx) = randi([params.class1PeriodMin, params.class1PeriodMax]);
            elseif p < params.probClass1 + params.probClass2
                devices.trafficClass(idx) = 2;
                devices.periodicInterval(idx) = 0;
            else
                devices.trafficClass(idx) = 3;
                devices.periodicInterval(idx) = 0;
            end

            idx = idx + 1;
        end
    end
end

function [x, y] = generateRandomPointInCell(centerX, centerY, radius)
    r = radius * sqrt(rand());
    theta = 2*pi*rand();

    x = centerX + r*cos(theta);
    y = centerY + r*sin(theta);
end

function resultsTable = runSimulation(params, gNB_positions, devices)
    maxRows = params.numDevices * params.numTimeSteps;
    numCols = 35;
    results = cell(maxRows, numCols);
    rowIdx = 1;

    for t = 1:params.numTimeSteps
        [isActive, packetSize_bytes, packetRate_pps, generatedTraffic_bps] = ...
            generateTrafficForTimeStep(devices, params, t);

        [distances_m, pathLoss_dB, rxPower_dBm] = ...
            computeRadioConditions(devices, gNB_positions, params);

        [servingRSRP_dBm, servingCell] = max(rxPower_dBm, [], 2);

        activeDevicesCell = zeros(params.numCells,1);
        totalAssociatedDevicesCell = zeros(params.numCells,1);

        for c = 1:params.numCells
            totalAssociatedDevicesCell(c) = sum(servingCell == c);
            activeDevicesCell(c) = sum((servingCell == c) & isActive);
        end

        activeDevicesTotal = sum(isActive);

        for i = 1:params.numDevices
            currentServingCell = servingCell(i);
            neighborCells = setdiff(1:params.numCells, currentServingCell);

            signal_dBm = rxPower_dBm(i, currentServingCell);
            signal_mW = dBm2mW(signal_dBm, params.minLinearPower_mW);

            interf_mW = 0;
            for k = 1:length(neighborCells)
                interf_mW = interf_mW + dBm2mW(rxPower_dBm(i, neighborCells(k)), params.minLinearPower_mW);
            end

            noise_mW = dBm2mW(params.thermalNoise_dBm, params.minLinearPower_mW);

            sinr_linear = signal_mW / max(interf_mW + noise_mW, params.minLinearPower_mW);
            sinr_dB = 10 * log10(max(sinr_linear, params.minLinearPower_mW));

            distServing = distances_m(i, currentServingCell);
            distNb1 = distances_m(i, neighborCells(1));
            distNb2 = distances_m(i, neighborCells(2));

            pathLossServing_dB = pathLoss_dB(i, currentServingCell);
            pathLossNb1_dB = pathLoss_dB(i, neighborCells(1));
            pathLossNb2_dB = pathLoss_dB(i, neighborCells(2));

            if totalAssociatedDevicesCell(currentServingCell) > 0
                cellLoad = activeDevicesCell(currentServingCell) / totalAssociatedDevicesCell(currentServingCell);
            else
                cellLoad = 0;
            end

            if isActive(i) && activeDevicesCell(currentServingCell) > 0
                allocatedBandwidth_Hz = params.bandwidth_Hz / activeDevicesCell(currentServingCell);
                spectralEfficiency_bpsHz = params.alphaEfficiency * log2(1 + sinr_linear);
                throughput_bps = allocatedBandwidth_Hz * spectralEfficiency_bpsHz;
                throughput_bps = min(throughput_bps, params.maxThroughput_bps);
            else
                allocatedBandwidth_Hz = 0;
                spectralEfficiency_bpsHz = 0;
                throughput_bps = 0;
            end

            [latency_ms, procDelay_ms, queueDelay_ms, txDelay_ms] = ...
                computeLatency(packetSize_bytes(i), throughput_bps, cellLoad, params);

            perVal = computePER(sinr_linear, cellLoad, packetSize_bytes(i), params);

            results(rowIdx, :) = {
                params.simulationName, ...
                params.version, ...
                t, ...
                devices.id(i), ...
                devices.homeCell(i), ...
                devices.trafficClass(i), ...
                devices.x(i), ...
                devices.y(i), ...
                currentServingCell, ...
                isActive(i), ...
                activeDevicesCell(currentServingCell), ...
                totalAssociatedDevicesCell(currentServingCell), ...
                activeDevicesTotal, ...
                cellLoad, ...
                distServing, ...
                distNb1, ...
                distNb2, ...
                pathLossServing_dB, ...
                pathLossNb1_dB, ...
                pathLossNb2_dB, ...
                packetSize_bytes(i), ...
                packetRate_pps(i), ...
                generatedTraffic_bps(i), ...
                allocatedBandwidth_Hz, ...
                servingRSRP_dBm(i), ...
                sinr_dB, ...
                throughput_bps, ...
                latency_ms, ...
                procDelay_ms, ...
                queueDelay_ms, ...
                txDelay_ms, ...
                interf_mW, ...
                noise_mW, ...
                spectralEfficiency_bpsHz, ...
                perVal ...
            };

            rowIdx = rowIdx + 1;
        end
    end

    results = results(1:rowIdx-1, :);

    varNames = {
        'simulation_name', ...
        'simulation_version', ...
        'time_step', ...
        'device_id', ...
        'home_cell', ...
        'traffic_class', ...
        'x_m', ...
        'y_m', ...
        'serving_cell', ...
        'is_active', ...
        'active_devices_serving_cell', ...
        'associated_devices_serving_cell', ...
        'active_devices_total', ...
        'cell_load', ...
        'distance_serving_m', ...
        'distance_neighbor1_m', ...
        'distance_neighbor2_m', ...
        'pathloss_serving_dB', ...
        'pathloss_neighbor1_dB', ...
        'pathloss_neighbor2_dB', ...
        'packet_size_bytes', ...
        'packet_rate_pps', ...
        'generated_traffic_bps', ...
        'allocated_bandwidth_Hz', ...
        'RSRP_dBm', ...
        'SINR_dB', ...
        'throughput_bps', ...
        'latency_ms', ...
        'proc_delay_ms', ...
        'queue_delay_ms', ...
        'tx_delay_ms', ...
        'interference_mW', ...
        'noise_mW', ...
        'spectral_efficiency_bpsHz', ...
        'PER' ...
    };

    resultsTable = cell2table(results, 'VariableNames', varNames);

% Ensure numeric variables have numeric type only if needed
numericVars = setdiff(varNames, {'simulation_name', 'simulation_version'});

for i = 1:length(numericVars)
    vn = numericVars{i};

    if iscell(resultsTable.(vn))
        resultsTable.(vn) = cell2mat(resultsTable.(vn));
    end
end

if iscell(resultsTable.simulation_name)
    resultsTable.simulation_name = string(resultsTable.simulation_name);
else
    resultsTable.simulation_name = string(resultsTable.simulation_name);
end

if iscell(resultsTable.simulation_version)
    resultsTable.simulation_version = string(resultsTable.simulation_version);
else
    resultsTable.simulation_version = string(resultsTable.simulation_version);
end
end

function [isActive, packetSize_bytes, packetRate_pps, generatedTraffic_bps] = ...
    generateTrafficForTimeStep(devices, params, t)

    N = length(devices.id);

    isActive = false(N,1);
    packetSize_bytes = zeros(N,1);
    packetRate_pps = zeros(N,1);
    generatedTraffic_bps = zeros(N,1);

    for i = 1:N
        cls = devices.trafficClass(i);

        switch cls
            case 1 % periodic
                interval = max(devices.periodicInterval(i), 1);
                isActive(i) = (mod(t, interval) == 0);

                packetSize_bytes(i) = randi([params.class1PacketMin, params.class1PacketMax]);
                packetRate_pps(i) = 1 / interval;

            case 2 % event-driven
                isActive(i) = (rand() < params.class2ActivationProb);

                packetSize_bytes(i) = randi([params.class2PacketMin, params.class2PacketMax]);
                packetRate_pps(i) = rand() * 0.5;

            case 3 % high-load
                isActive(i) = (rand() < params.class3ActivationProb);

                packetSize_bytes(i) = randi([params.class3PacketMin, params.class3PacketMax]);
                packetRate_pps(i) = rand() * 2 + 0.5;

            otherwise
                isActive(i) = false;
                packetSize_bytes(i) = 0;
                packetRate_pps(i) = 0;
        end

        generatedTraffic_bps(i) = packetSize_bytes(i) * 8 * packetRate_pps(i);
    end
end

function [distances_m, pathLoss_dB, rxPower_dBm] = ...
    computeRadioConditions(devices, gNB_positions, params)

    N = length(devices.id);
    C = size(gNB_positions, 1);

    distances_m = zeros(N, C);
    pathLoss_dB = zeros(N, C);
    rxPower_dBm = zeros(N, C);

    freq_MHz = params.carrierFreq_GHz * 1000;

    for i = 1:N
        for c = 1:C
            dx = devices.x(i) - gNB_positions(c,1);
            dy = devices.y(i) - gNB_positions(c,2);

            d = sqrt(dx^2 + dy^2);
            d = max(d, params.minDistance_m);

            distances_m(i,c) = d;

            % Simplified large-scale path loss model
            % PL(dB) = 32.4 + 20log10(f_MHz) + 10*n*log10(d_km)
            d_km = d / 1000;
            PL_dB = 32.4 + 20*log10(freq_MHz) + 10*params.pathLossExponent*log10(d_km);

            shadowing_dB = params.shadowStdDev_dB * randn();

            pathLoss_dB(i,c) = PL_dB - shadowing_dB;
            rxPower_dBm(i,c) = params.txPower_dBm - pathLoss_dB(i,c);
        end
    end
end

function [latency_ms, procDelay_ms, queueDelay_ms, txDelay_ms] = ...
    computeLatency(packetSize_bytes, throughput_bps, cellLoad, params)

    procDelay_ms = params.procDelayMin_ms + ...
        (params.procDelayMax_ms - params.procDelayMin_ms) * rand();

    queueDelay_ms = params.baseQueueDelay_ms + ...
        params.queueLoadFactor_ms * cellLoad + ...
        params.queueRandomness_ms * rand();

    if throughput_bps > 0
        txDelay_ms = ((packetSize_bytes * 8) / throughput_bps) * 1000;
    else
        txDelay_ms = params.inactivePenaltyTxDelay_ms;
    end

    latency_ms = procDelay_ms + queueDelay_ms + txDelay_ms;
end

function perVal = computePER(sinr_linear, cellLoad, packetSize_bytes, params)
    normPacket = packetSize_bytes / params.class3PacketMax;

    perVal = exp(-params.perSinrFactor * max(sinr_linear, 1e-9)) + ...
             params.perLoadFactor * cellLoad + ...
             params.perPacketFactor * normPacket;

    perVal = min(max(perVal, 0), 1);
end

function mW = dBm2mW(dBm, floorValue)
    mW = 10^(dBm/10);
    mW = max(mW, floorValue);
end

function writeOutputs(resultsTable, params)
    writetable(resultsTable, params.outputCsvFile);

    fid = fopen(params.outputSummaryFile, 'w');

    fprintf(fid, '=== PRIVATE 5G IoT SIMULATION SUMMARY ===\n');
    fprintf(fid, 'Simulation Name: %s\n', params.simulationName);
    fprintf(fid, 'Version: %s\n', params.version);
    fprintf(fid, 'Cells: %d\n', params.numCells);
    fprintf(fid, 'Devices: %d\n', params.numDevices);
    fprintf(fid, 'Time Steps: %d\n', params.numTimeSteps);
    fprintf(fid, 'Carrier Frequency [GHz]: %.2f\n', params.carrierFreq_GHz);
    fprintf(fid, 'Bandwidth per Cell [Hz]: %.0f\n', params.bandwidth_Hz);
    fprintf(fid, 'Tx Power [dBm]: %.2f\n', params.txPower_dBm);
    fprintf(fid, 'Thermal Noise [dBm]: %.2f\n', params.thermalNoise_dBm);
    fprintf(fid, '\n');

    fprintf(fid, '=== KPI SUMMARY ===\n');
    fprintf(fid, 'Mean RSRP [dBm]: %.4f\n', mean(resultsTable.RSRP_dBm));
    fprintf(fid, 'Std RSRP [dBm]: %.4f\n', std(resultsTable.RSRP_dBm));
    fprintf(fid, 'Mean SINR [dB]: %.4f\n', mean(resultsTable.SINR_dB));
    fprintf(fid, 'Std SINR [dB]: %.4f\n', std(resultsTable.SINR_dB));
    fprintf(fid, 'Mean Throughput [bps]: %.4f\n', mean(resultsTable.throughput_bps));
    fprintf(fid, 'Std Throughput [bps]: %.4f\n', std(resultsTable.throughput_bps));
    fprintf(fid, 'Mean Latency [ms]: %.4f\n', mean(resultsTable.latency_ms));
    fprintf(fid, 'Std Latency [ms]: %.4f\n', std(resultsTable.latency_ms));
    fprintf(fid, 'Mean PER: %.6f\n', mean(resultsTable.PER));
    fprintf(fid, 'Std PER: %.6f\n', std(resultsTable.PER));
    fprintf(fid, '\n');

    fprintf(fid, '=== ACTIVITY SUMMARY ===\n');
    fprintf(fid, 'Active ratio: %.4f\n', mean(resultsTable.is_active));
    fprintf(fid, 'Average cell load: %.4f\n', mean(resultsTable.cell_load));
    fprintf(fid, '\n');

    fprintf(fid, '=== TRAFFIC CLASS COUNTS ===\n');
    for cls = 1:3
        fprintf(fid, 'Class %d rows: %d\n', cls, sum(resultsTable.traffic_class == cls));
    end

    fclose(fid);
end