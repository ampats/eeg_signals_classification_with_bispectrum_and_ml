%plots signal over time, fft, bispectrum and bispectrum with common color
%limits diagrams for 1 subject (subNum) and for 1 task (sesNum), for all
%ear-EEG channels

run('load_preprocessed_data_plots.m') ;
x = [0:L-1]/fs ; 

for i=1:14 %14 ear-EEG channels
    
    %plot signal over time
    figure(i)
    plot(x,dataNorm(:,i)) ;
    title({'Signal of channel',char(chan(i+32)), 'over time (subject ', subNum, ' - running)'})
    xlabel('time(s)')
    saveas(gcf, sprintf('new_CH%d_s1_v2.png',i))
    
    %plot fft
    y = fft(dataNorm(:,i)) ;
    p2 = abs(y/L) ;
    p1 = p2(1:L/2+1) ;
    p1(2:end-1) = 2*p1(2:end-1);
    
    figure(i+14)
    f = fs*(0:(L/2))/L;
    plot(f,p1) 
    title({'Single-Sided Amplitude Spectrum of signal of channel ',char(chan(i+32)),' (subject ',subNum, ' - running)'})
    xlabel('f (Hz)')
    ylabel('|P1(f)|')
    saveas(gcf, sprintf('new_CH%d_s1_v2_fft.png',i))

    %plot signal bispectrum
    figure(i+28)
    [bd,waxis] = bispecd(dataNorm(:,i),512,5,512,50); 
    title({'Bispectrum estimated via the direct (FFT) method for signal of channel ',char(chan(i+32)),' (subject ',subNum, '- running)'})
    colorbar
    saveas(gcf, sprintf('NEWnew_CH%d_s1_v5_bispec.png',i))
    
    %plot signal bispectrum with common color limits
    figure(i+42)
    [bd,waxis] = bispecd_ccl(dataNorm(:,i),512,5,512,50); 
    title({'Bispectrum (with common color limits for all samples) for signal of channel ',char(chan(i+32)),' (subject ',subNum, '- standing)'})
    colorbar
    saveas(gcf, sprintf('new_CH%d_s1_v2_bispec_ccl.png',i))
    
end