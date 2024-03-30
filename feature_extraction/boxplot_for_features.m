% generates the boxplots for each of the 40 features
% each diagram contains 3 boxplots for each of the 3 classes and for 1 feature

data = csvread('data_no_bad_chans.csv') ; %replace with 'data_no_bad_chans_reRef' to use the data with contralateral-mean re-ref method

% boxplot for features
featureLabels = ["BispecPeak", "Freq of BispecPeak", "WCOB - f1m", "WCOB - f2m", "NBE", "NBSE", "J", "Skewness", "Kurtosis", "Variance", "MMOB", "Bispec-magnitude variability", "SOLA", "SOLADE_1", "SOLADE_2", "SOLAHE_3", "FOSM_1", "FOSM_2", "FOSM_3", "SOSM_1", "SOSM_2", "SOSM_3", "SumOfAmplitudesDE_1", "SumOfAmplitudesDE_2", "SumOfAmplitudesHE_3", "SSI_1", "SSI_2", "SSI_3", "RootMeanSquareDE_1", "RootMeanSquareDE_2", "RootMeanSquareHE_3", "VarDE_1", "VarDE_2", "VarHE_3", "V3orderDE_1", "V3orderDE_2", "V3orderHE_3", "LogDE_1", "LogDE_2", "LogHE_3"] ;
for i=1:40
    figure(i) 
    boxplot(data(:,i), data(:,41))
    title(featureLabels(i))
    saveas(gcf, sprintf('feature %i.png',i))
end