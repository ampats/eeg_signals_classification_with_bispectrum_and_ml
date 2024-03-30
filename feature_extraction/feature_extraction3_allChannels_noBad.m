% feature extraction of data for all good channels (9), for subjects 1-15 and for 3 classes "standing", "slow walking" and "running"

%load data
run('load_preprocessed_data3_allChannels.m') ; %'load_reRef_data3ALL.m' for contralateral-mean reRef data

%feature matrix
nFeat = 40 ;
features = zeros(nSub*3*nChan,nFeat) ;
feat_y_sort = zeros(nSub*3, nFeat+1) ;
counter = 0 ;
samplIndex = 1 ;

%starting data point for each sample, each point corresponds to the starting point of a trial 
%s1ses2, s1ses3, s1ses5, s2ses2, s2ses3,  s2ses5, ...
samplNum = [1034 6139 6955 12035 6168 1859 2762 33124 4252 2881 6165 9605 948 6155 5283 10478 6154 4096 14900 6082 32237 12138 6018 5144 12083 6017 38932 31489 6159 14989 8252 6005 4970 6147 14139 17600 12034 6060 5242 14071 14138 10032 5987 14126 9897] ;

for k = 1:nSub*3*nChan
    
    %bispectrum estimation
    [bd,waxis] = bispecd(dataNorm(:,k),512,5,512,50); 
    
    % 1. bispectrum peak
    maxCol = max(abs(bd)) ;
    features(k,1) = max(maxCol) ;
    
    % 2. frequency of bispectrum peak
    abs_bd = abs(bd) ;
    % Find the peak in the bispectrum
    [~, ind] = max(abs_bd(:));
    [row, col] = ind2sub(size(abs_bd), ind);
    % Extract the frequency associated with the peak
    peak_frequency = waxis(row);
    features(k,2) = abs(peak_frequency) ;

    % 3.Weighted Center of Bispectrum (WCOB)
    sum1 = 0 ;
    sumx = 0 ;
    sumy = 0 ;
    
    %old Omega region
%     bd_omega = zeros(256) ;
%     f_omega = zeros(256,256,2) ;
%     % getting points from non-redundant region of bispectrum, Omega
%     for i=257:512
%         for j=257:512
%             if waxis(j)<=waxis(i) && waxis(j)<=50-waxis(i)
%                 bd_omega(i-256,j-256) = abs(bd(i,j)) ;
%                 f_omega(i-256,j-256,1) = waxis(i) ;
%                 f_omega(i-256,j-256,2) = waxis(j) ;
%                 sum1 = sum1 + abs(bd(i,j)) ;
%                 sumx = sumx + waxis(i)*abs(bd(i,j)) ;
%                 sumy = sumy + waxis(j)*abs(bd(i,j)) ;
%             end
%         end
%     end

    % getting points from non-redundant region of bispectrum, Omega
    bd_omega = zeros(256,512) ; %change: 256 instead of 257
    f_omega = zeros(256,512,2) ;
    for j=1:512 %f1
       for i=1:256 %f2
           if (waxis(j)<=0 && waxis(i)<=waxis(j)) || (waxis(j)>=0 && waxis(i)<=-waxis(j))   %waxis(i)<=waxis(j) &&
                bd_omega(i,j) = abs(bd(i,j)) ;
                f_omega(i,j,1) = waxis(j) ;
                f_omega(i,j,2) = waxis(i) ;
                sum1 = sum1 + abs(bd(i,j)) ;
                sumx = sumx + waxis(j)*abs(bd(i,j)) ;
                sumy = sumy + waxis(i)*abs(bd(i,j)) ;
            end
        end
    end

    features(k,3) = sumx/sum1 ; %f1m
    features(k,4) = sumy/sum1 ; %f2m
    f_omega_resh = reshape(f_omega,[512*256,2]) ;
    
    % 4.Normalized bispectral entropy (NBE)
    p = bd_omega/sum1 ;
    logp = zeros(256,512) ; 
    for j = 1:512
        for i = 1:256
            if p(i,j) ~= 0
                logp(i,j) = log2(p(i,j));
            end
        end
    end
    prod = p.*logp ;
    sumEnt = 0 ;
    for j = 1:512
        for i = 1:256
                sumEnt = sumEnt + prod(i,j) ;
        end
    end
    features(k,5) = -sumEnt ; %NBE

    % 5.Normalized Bispectral Squared Entropy (NBSE)
    sum2 = 0 ;
    % getting points from non-redundant region of bispectrum, Omega
    for j=1:512
        for i=1:256
            if (waxis(j)<=0 && waxis(i)<=waxis(j)) || (waxis(j)>=0 && waxis(i)<=-waxis(j))
                bd_omega(i,j) = abs(bd(i,j)) ;
                sum2 = sum2 + (abs(bd(i,j)))^2 ;
            end
        end
    end
    pn = bd_omega.^2/sum2 ;
    logpn = zeros(256,512) ; 
    for j = 1:512
        for i= 1:256
            if p(i,j) ~= 0
                logpn(i,j) = log2(pn(i,j));
            end
        end
    end
    prod = pn.*logpn ;
    sumEnt2 = 0 ;
    for j = 1:512
        for i = 1:256
                sumEnt2 = sumEnt2 + prod(i,j) ;
        end
    end
    features(k,6) = -sumEnt2 ; %NBSE

    % 6.Negentropy (J)
    features(k,7) = (1/12)*(skewness(dataNorm(:,k)))^2 + (1/48)*(kurtosis(dataNorm(:,k)))^2 ; %J

    % 7.Skewness 
    features(k,8) = skewness(dataNorm(:,k)) ;

    % 8.Kurtosis
    features(k,9) = kurtosis(dataNorm(:,k)) ;

    % 9.Variance
    counter = counter + 1 ;
    if counter >=1 && counter <=9
        features(k,10) = var(data_reshaped(k,samplNum(samplIndex):samplNum(samplIndex)+511)) ; 
    end
    if counter == 9 
        counter = 0 ;
        samplIndex = samplIndex+1 ;
    end
        
    % 10.Mean magnitude of Bispectrum (MMOB)
    L = size(bd_omega,1)*size(bd_omega,2) ;
    features(k,11) = (1/L)*sum1 ;
    
    % 11.Bispectrum-magnitude variability
    MMOB =  features(k,11) ;
    bdOmega_MMOB = bd_omega - MMOB ;
    features(k,12) = (1/L)*sum(bdOmega_MMOB, 'all') ;
    
    % 12.Sum of logarithmic amplitudes of Bispectrum (SOLA)
    logBd = zeros(256,512) ; 
    for j = 1:512
        for i = 1:256
            if bd_omega(i,j) ~= 0
                logBd(i,j) = log2(bd_omega(i,j));
            end
        end
    end
   features(k,13) = sum(logBd, 'all') ;
   
   % 13.Sum of logarithmic amplitudes of diagonal elements d1 (SOLADE)
   bd_omega_1 = zeros(256,1) ; 
   d1 = zeros(257,2) ;
   logBd_1 = zeros(256,1) ; 
   for j = 1:512
       for i = 1:256
           if i==j && bd_omega(i,j) ~= 0
               d1(i,1) = f_omega(i,j,1) ;
               d1(i,2) = f_omega(i,j,2) ;
               bd_omega_1(i) = bd_omega(i,j);
               logBd_1(i) = log2(bd_omega(i,j));
           end
       end
   end
   features(k,14) = sum(logBd_1) ;
   
   % 14.Sum of logarithmic amplitudes of diagonal elements d2 (SOLADE)
   bd_omega_2 = zeros(256,1) ; 
   d2 = zeros(256,2) ;
   logBd_2 = zeros(256,1) ; 
   for j = 1:512
       for i = 1:256
           if i+j==512 && bd_omega(i,j) ~= 0
               d2(i,1) = f_omega(i,j,1) ;
               d2(i,2) = f_omega(i,j,2) ;
               bd_omega_2(i) = bd_omega(i,j);
               logBd_2(i) = log2(bd_omega(i,j));
           end
       end
   end
   features(k,15) = sum(logBd_2) ;
   
   % 15.Sum of logarithmic amplitudes of height elements d3 (SOLADE)
   bd_omega_3 = zeros(256,1) ; 
   d3 = zeros(256,2) ;
   logBd_3 = zeros(256,1) ; 
   for i = 1:256
       d3(i,1) = f_omega(i,255,1) ;
       d3(j,2) = f_omega(i,255,2) ;
       bd_omega_3(i) = bd_omega(i,256);
       logBd_3(i) = log2(bd_omega(i,256));   
   end
   features(k,16) = sum(logBd_3) ;
   
  % 16.First order Spectral Moment of d1(FOSM) 
  n = 1:256 ;
  features(k,17) = n*logBd_1 ;
  
  % 17.First order Spectral Moment of d2(FOSM) 
  features(k,18) = n*logBd_2 ;
  
  % 18.First order Spectral Moment of d3(FOSM) 
  features(k,19) = n*logBd_3 ;
  
  % 19.Second order Spectral Moment of d1 (SOSM)
  fosm_1 = features(k,17) ;
  n2 = (n-fosm_1).^2 ;
  features(k,20) = n2*logBd_1 ;
  
  % 20.Second order Spectral Moment of d2 (SOSM)
  fosm_2 = features(k,18) ;
  n2 = (n-fosm_2).^2 ;
  features(k,21) = n2*logBd_2 ;
  
  % 21.Second order Spectral Moment of d3 (SOSM)
  fosm_3 = features(k,19) ;
  n2 = (n-fosm_3).^2 ;
  features(k,22) = n2*logBd_3 ;
  
  % 22.Sum of amplitudes of diagonal elements of d1
  features(k,23) = sum(bd_omega_1) ;
  
  % 23.Sum of amplitudes of diagonal elements of d2
  features(k,24) = sum(bd_omega_2) ;
  
  % 24.Sum of amplitudes of height elements of d3
  features(k,25) = sum(bd_omega_3) ;
  
  % 25.Simple Square Integral (SSI) of d1
  features(k,26) = sum(bd_omega_1.^2) ;
  
  % 26.Simple Square Integral (SSI) of d2
  features(k,27) = sum(bd_omega_2.^2) ;
  
  % 27.Simple Square Integral (SSI) of d3
  features(k,28) = sum(bd_omega_3.^2) ;
  
  % 28.Root mean square of d1 diagonal elements
  N = size(n,2) ;
  ssi_1 = features(k,26) ;
  features(k,29) = sqrt((1/N)*ssi_1) ;
  
  % 29.Root mean square of d2 diagonal elements
  ssi_2 = features(k,27) ;
  features(k,30) = sqrt((1/N)*ssi_2) ;
  
  % 30.Root mean square of d3 height elements
  ssi_3 = features(k,28) ;
  features(k,31) = sqrt((1/N)*ssi_3) ;
  
  % 31.Variance (VAR) of d1
  features(k,32) = (1/(N-1))*ssi_1 ;
  
  % 32.Variance (VAR) of d2
  features(k,33) = (1/(N-1))*ssi_2 ;
  
  % 33.Variance (VAR) of d3
  features(k,34) = (1/(N-1))*ssi_3 ;
  
  % 34.V3 order of d1
  features(k,35) = ((1/N)*sum(bd_omega_1.^3))^(1/3) ;
  
  % 35.V3 order of d2
  features(k,36) = ((1/N)*sum(bd_omega_2.^3))^(1/3) ;
  
  % 36.V3 order of d3
  features(k,37) = ((1/N)*sum(bd_omega_3.^3))^(1/3) ;
  
  % 37.Log Detector (LOG) of d1
  features(k,38) = exp((1/N)*sum(logBd_1)) ;
  
  % 38.Log Detector (LOG) of d2
  features(k,39) = exp((1/N)*sum(logBd_2)) ;
  
  % 39.Log Detector (LOG) of d3
  features(k,40) = exp((1/N)*sum(logBd_3)) ;
  
end

%label 0: standing
y = zeros(nSub*3*nChan,1);
%label 1: slow walking
for i=10:27:nSub*3*nChan
    y(i:i+8) = 1 ;
end 
%label 2: running
for i=19:27:nSub*3*nChan
    y(i:i+8) = 2 ;
end 
%features_y = [features(:,[1:11,13:23,26,29,32,35,38]) y] ;
%features_y = sortrows(features_y,28) ;
features_y = [features y] ;
features_y_sorted = sortrows(features_y,41) ;
csvwrite('data_no_bad_chans.csv', features_y_sorted) ; % 'data_no_bad_chans_reRef' for contralateral-mean re-ref data

% run to keep data of only one channel, depending on start index of for loop
% start index 1-9 corresponds to chans 1-9 
for k=1:9
    j = 1 ;
    for i=k:9:size(features_y_sorted,1)
        feat_y_sort(j,:) = features_y_sorted(i,:) ;
        j = j+1 ;
    end
    csvwrite(sprintf('data3_noBad_chan%i.csv',k), feat_y_sort) ;
end