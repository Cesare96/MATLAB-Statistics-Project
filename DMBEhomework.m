%%
%%%%ESERCIZIO 1%%%%%%%% 


table1=readtable('gdpdeu.csv');
table2=readtable('gdpfr.csv');
table3=readtable('gdpit.csv');
table4=readtable('gdpesp.csv');
time=table2array(table1(:,1));

gdp_ge=table2array(table1(:,2));
gdp_fr=table2array(table2(:,2));
gdp_it=table2array(table3(:,2));
gdp_esp=table2array(table4(:,2));
%%
%%%%CHANGES AND RATE OF CHANGES%%%%
gdp_chgge=diff(gdp_ge);
gdp_chgfr=diff(gdp_fr);
gdp_chgit=diff(gdp_it);
gdp_chgsp=diff(gdp_esp);


roc_ge=gdp_chgge./gdp_ge(1:end-1)*100;
roc_fr=gdp_chgfr./gdp_fr(1:end-1)*100;
roc_it=gdp_chgit./gdp_it(1:end-1)*100;
roc_esp=gdp_chgsp./gdp_esp(1:end-1)*100;

%%
%%%%%PLOTTING DATA%%%%%%

%%plot changes
figure
subplot(2,2,1)
plot(time(1:end-1),gdp_chgge,'-*')
xlabel('years')
ylabel('Millions of chained euros')
title(' German annual GDP, millions of chained 2010 euros,differences')
subplot(2,2,2)
plot(time(1:end-1),gdp_chgfr,'-*')
xlabel('years')
ylabel('Millions of chained euros')
title(' French annual GDP, millions of chained 2010 euros,differences')
subplot(2,2,3)
plot(time(1:end-1),gdp_chgit,'-*')
xlabel('years')
ylabel('Millions of chained euros')
title(' Italian annual GDP, millions of chained 2010 euros,differences')
subplot(2,2,4)
plot(time(1:end-1),gdp_chgsp,'-*')
xlabel('years')
ylabel('Millions of chained euros')
title(' Spanish annual GDP, millions of chained euros,differences')
%%
%%plot rate of changes
figure
subplot(2,2,1)
plot(time(1:end-1),roc_ge,'-*')
xlabel('years')
ylabel('%')
title('German GDP annual growth rate')
subplot(2,2,2)
plot(time(1:end-1),roc_fr,'-*')
xlabel('years')
ylabel('%')
title('French GDP annual growth rate')
subplot(2,2,3)
plot(time(1:end-1),roc_it,'-*')
xlabel('years')
ylabel('%')
title('italian GDP annual growth rate')
subplot(2,2,4)
plot(time(1:end-1),roc_esp,'-*')
xlabel('years')
ylabel('%')
title('Spanish GDP annual growth rate')

%%
%%%LOCATION OF CHANGES 
meanGEchg=mean(gdp_chgge);
meanFRchg=mean(gdp_chgfr);
meanITchg=mean(gdp_chgit);
meanESPchg=mean(gdp_chgsp);


medianGEchg=median(gdp_chgge);
medianFRchg=median(gdp_chgfr);
medianITchg=median(gdp_chgit);
medianESPchg=median(gdp_chgsp);

modeGEchg=mode(gdp_chgge);
modeFRchg=mode(gdp_chgfr);
modeITchg=mode(gdp_chgit);
modeESPchg=mode(gdp_chgsp);
%%
% % %LOCATION RATE OF CHANGE
meanGEroc=mean(roc_ge);
meanFRroc=mean(roc_fr);
meanITroc=mean(roc_it);
meanESProc=mean(roc_esp);


medianGEroc=median(roc_ge);
medianFRroc=median(roc_fr);
medianITroc=median(roc_it);
medianESProc=median(roc_esp);

modeGEroc=mode(roc_ge);
modeFRroc=mode(roc_fr);
modeITroc=mode(roc_it);
modeESProc=mode(roc_esp);
%%

%%%DISPERSION OF CHANGES
RangeGEchg=max(gdp_chgge)-min(gdp_chgge);
RangeFRchg=max(gdp_chgfr)-min(gdp_chgfr);
RangeITchg=max(gdp_chgit)-min(gdp_chgit);
RangeESPchg=max(gdp_chgsp)-min(gdp_chgsp);

IqrGEchg=quantile(gdp_chgge,0.75)-quantile(gdp_chgge,0.25);
IqrFRchg=quantile(gdp_chgfr,0.75)-quantile(gdp_chgfr,0.25);
IqrITchg=quantile(gdp_chgit,0.75)-quantile(gdp_chgit,0.25);
IqrESPchg=quantile(gdp_chgsp,0.75)-quantile(gdp_chgsp,0.25);


figure
boxplot([gdp_chgge,gdp_chgfr,gdp_chgit,gdp_chgsp],'labels',{'GE','FR','ITA','ESP'})
title('five number summary-changes')


varGEchg=var(gdp_chgge);
varFRchg=var(gdp_chgfr);
varITchg=var(gdp_chgit);
varESPchg=var(gdp_chgsp);


stdGEchg=std(gdp_chgge);
stdFRchg=std(gdp_chgfr);
stdITchg=std(gdp_chgit);
stdESPchg=std(gdp_chgsp);
%%
% % % DISPERSION RATE OF CHANGES

RangeGEroc=max(roc_ge)-min(roc_ge);
RangeFRroc=max(roc_fr)-min(roc_fr);
RangeITroc=max(roc_it)-min(roc_it);
RangeESProc=max(roc_esp)-min(roc_esp);

IqrGEroc=quantile(roc_ge,0.75)-quantile(roc_ge,0.25);
IqrFRroc=quantile(roc_fr,0.75)-quantile(roc_fr,0.25);
IqrITroc=quantile(roc_it,0.75)-quantile(roc_it,0.25);
IqrESProc=quantile(roc_esp,0.75)-quantile(roc_esp,0.25);


figure
boxplot([roc_ge,roc_fr,roc_it,roc_esp],'labels',{'GE','FR','ITA','ESP'})
title('five number summary-rate of change')


varGEroc=var(roc_ge);
varFRroc=var(roc_fr);
varITroc=var(roc_it);
varESProc=var(roc_esp);


stdGEroc=std(roc_ge);
stdFRroc=std(roc_fr);
stdITroc=std(roc_it);
stdESProc=std(roc_esp);
%%

%%%%%DEPENDENCY CHANGES
covGE_FR=cov(gdp_chgge,gdp_chgfr);
covGE_IT=cov(gdp_chgge,gdp_chgit);
covGE_ESP=cov(gdp_chgge,gdp_chgsp);
covFR_IT=cov(gdp_chgfr,gdp_chgit);
covFR_ESP=cov(gdp_chgfr,gdp_chgsp);
covIT_ESP=cov(gdp_chgit,gdp_chgsp);


rhoGE_FR=corr(gdp_chgge,gdp_chgfr);
rhoGE_IT=corr(gdp_chgge,gdp_chgit);
rhoGE_ESP=corr(gdp_chgge,gdp_chgsp);
rhoFR_IT=corr(gdp_chgfr,gdp_chgit);
rhoFR_ESP=corr(gdp_chgfr,gdp_chgsp);
rhoIT_ESP=corr(gdp_chgit,gdp_chgsp);
%%
% % % DEPENDENCY RATEs OF CHANGE
covGE_FRroc=cov(roc_ge,roc_fr);
covGE_ITroc=cov(roc_ge,roc_it);
covGE_ESProc=cov(roc_ge,roc_esp);
covFR_ITroc=cov(roc_fr,roc_it);
covFR_ESProc=cov(roc_fr,roc_esp);
covIT_ESProc=cov(roc_it,roc_esp);


rhoGE_FRroc=corr(roc_ge,roc_fr);
rhoGE_ITroc=corr(roc_ge,roc_it);
rhoGE_ESProc=corr(roc_ge,roc_esp);
rhoFR_ITroc=corr(roc_fr,roc_it);
rhoFR_ESProc=corr(roc_fr,roc_esp);
rhoIT_ESProc=corr(roc_it,roc_esp);
%%
% % % SCATTER PLOT CHANGES
figure
plot(gdp_chgge,gdp_chgfr,'*')
xlabel('germany')
ylabel('france')
lsline
figure
plot(gdp_chgge,gdp_chgit,'*')
xlabel('germany')
ylabel('italy')
lsline
figure
plot(gdp_chgge,gdp_chgsp,'*')
xlabel('germany')
ylabel('spain')
lsline
figure
plot(gdp_chgfr,gdp_chgit,'*')
xlabel('france')
ylabel('italy')
lsline
figure
plot(gdp_chgfr,gdp_chgsp,'*')
xlabel('france')
ylabel('spain')
lsline
figure
plot(gdp_chgit,gdp_chgsp,'*')
xlabel('italy')
ylabel('spain')
lsline
%%
% % SCATTER PLOT RATES OF CHANGE
figure
plot(roc_ge,roc_fr,'r*')
xlabel('germany')
ylabel('france')
lsline
figure
plot(roc_ge,roc_it,'b*')
xlabel('germany')
ylabel('italy')
lsline
figure
plot(roc_ge,roc_esp,'c*')
xlabel('germany')
ylabel('spain')
lsline
figure
plot(roc_fr,roc_it,'k*')
xlabel('france')
ylabel('italy')
lsline
figure
plot(roc_fr,roc_esp,'*')
xlabel('france')
ylabel('spain')
lsline
figure
plot(roc_it,roc_esp,'*')
xlabel('italy')
ylabel('spain')
lsline
%% confidence interval

vettoreroc=[roc_ge,roc_fr,roc_it,roc_esp];
vettorenomi={'roc germany','roc france','roc italy','roc spain'};
p=[0.90,0.95,0.99];
uci=nan(1,length(p));
lci=nan(1:length(p));
xn=nan;
figure
for i=1:4
 n=length(vettoreroc(:,i));
 xn(i)=mean(vettoreroc(:,i));
 sigman=(std(vettoreroc(:,i)));
 for k=1:length(p)
     alpha=1-p(k);
     t=tinv(p(k)+alpha/2,n-1);
     uci(k)=xn(i)+(sigman/sqrt(n))*t;
     lci(k)=xn(i)-(sigman/sqrt(n))*t;
 end

     subplot(2,2,i)
     plot([lci(1),uci(1)],[1.5,1.5])
     hold on 
     plot([lci(2),uci(2)],[2,2])
     hold on
     plot([lci(3),uci(3)],[2.5,2.5])
     hold on
     plot(xn(i),1.5,'o',xn(i),2,'o',xn(i),2.5,'o','MarkerSize',5,'MarkerEdgeColor','red','MarkerFaceColor','red')
     xlim([-1.5,4])
     ylim([0,3])
     title('confidence interval',vettorenomi(i))
     xtickformat('percentage')
     legend('0.90','0.95','0.99')
end
%% hypothesis test 
vettorehypothesistestcountryname={'htgermany','htfrance','htitaly','htspain'};
vettoremedieroc=[meanGEroc,meanFRroc,meanITroc,meanESProc];
vettorestdroc=[stdGEroc,stdFRroc,stdITroc,stdESProc];
t_90=tinv(0.90+0.1/2,24);
t_95=tinv(0.95+0.05/2,24);
t_99=tinv(0.99+0.01/2,24);
t_critico=[t_90,t_95,t_99];
t_n=(0);

for i=1:length(vettorehypothesistestcountryname)
    vettorehypothesistestcountryname(i)
    t_n(i)=vettoremedieroc(i)/(vettorestdroc(i)/sqrt(25));
    for k=1:3
      if abs(t_n(i))>t_critico(k) 
       disp('reject the null hypothesis')
      else 
         disp('do not reject the null hypothesis')
      end
    end
end
%% hypothesis test with p-value

vettorehypothesistestcountryname_p_value={'htgermany_p-value','htfrance_p-value','htitaly_p-value','htspain_p-value'};
critical_value=[0.1,0.05,0.01];
p_value=zeros(1,4);
for i =1:length(vettorehypothesistestcountryname_p_value)
    vettorehypothesistestcountryname_p_value(i)
p_value(i)=2*tcdf(-abs(t_n(i)),24);
 for k= 1:3
     if p_value(i)<critical_value(k)
         disp('reject the null hypothesis')
     else ;disp('do not reject the null hypothesis')
     end
 end
end
%% exercise 2

% % roll rigged die

p1 = 2/15;
p2 = 1/3;
p3 = 2/15;
p4 = 2/15;
p5 = 2/15;
p6 = 2/15;

q = [p1, p2, p3, p4, p5,p6];

partition = [0,cumsum(q)];
%%
%%%richiesta a)

%%%population mean
v=(1:length(q));
riggedvecpopmean=v.*q;
riggedpopmean=sum(riggedvecpopmean);

%%%population variance
riggedvecpopvar=((v-riggedpopmean).^2).*q;
riggedpopvar=sum(riggedvecpopvar);
%%
%%%richiesta b)
%%%one roll of a rigged dice
u = rand;
if u>partition(1) && u<partition(2)
        dice = 1;
    elseif u>partition(2) && u<partition(3)
        dice = 2;
    elseif u>partition(3) && u<partition(4)
        dice = 3;
    elseif u>partition(4) && u<partition(5)
        dice = 4;
    elseif u>partition(5) && u<partition(6)
        dice = 5;
    elseif u>partition(6) && u<partition(7)
        dice = 6;
end

fprintf('result of roll of a rigged dice is: %f \n', dice)

%%
%%%rolling lots of rigged dice
%%%barchart frequency rigged dice
yaxis = zeros([1,1000]);
figure
for i = 1:1000
    u = rand;
    
    if u>partition(1) && u<partition(2)
        dice = 1;
    elseif u>partition(2) && u<partition(3)
        dice = 2;
    elseif u>partition(3) && u<partition(4)
        dice = 3;
    elseif u>partition(4) && u<partition(5)
        dice = 4;
    elseif u>partition(5) && u<partition(6)
        dice = 5;
    elseif u>partition(6) && u<partition(7)
        dice = 6;
    end
    yaxis(i) = dice;
end
[frequency, values]  = groupcounts(yaxis');
xaxis = (frequency'./(sum(frequency))).*100;
transbin = values';
bar(transbin,xaxis);
title('frequency distribution of a simulated rigged dice');
%% demotrating conrvengence to the mean(weak law of large numbers)
figure
yax = (0);
samplemean = (0);
xax = (0);
popmean = (0);

for i = 1:1000
    u = rand;
    if u>partition(1) && u<partition(2)
        dice = 1;
    elseif u>partition(2) && u<partition(3)
        dice = 2;
    elseif u>partition(3) && u<partition(4)
        dice = 3;
    elseif u>partition(4) && u<partition(5)
        dice = 4;
    elseif u>partition(5) && u<partition(6)
        dice = 5;
    elseif u>partition(6) && u<partition(7)
        dice = 6;
    end
    yax(i) = dice;
    samplemean(i) = mean(yax);
    xax(i) = i;
    popmean(i) = 3.2;
    plot(xax, samplemean, xax, popmean)
    legend('sample mean','population mean')
    drawnow;
end
% % % the assumption is that the random variables must be iid
%% exercise 3
% % % richiesta a

fp1 = 1/6;
fp2 = 1/6;
fp3 = 1/6;
fp4 = 1/6;
fp5 = 1/6;
fp6 = 1/6;

fq = [fp1, fp2, fp3, fp4, fp5,fp6];

fpartition = [0,cumsum(fq)];
vettoredado=(1:6);
fairprobvec=[fp1,fp2,fp3,fp4,fp5,fp6];
fairpopvec=fairprobvec.*vettoredado;
fairpopmean=sum(fairpopvec);
fairpopvarvec=((vettoredado-fairpopmean).^2).*fairprobvec;
fairpopvar=sum(fairpopvarvec);
fairpopstd=sqrt(fairpopvar);
%% roll of a fair dice
fu = rand;

if fu>fpartition(1) && fu<fpartition(2)
        fdice = 1;
elseif fu>fpartition(2) && fu<fpartition(3)
        fdice = 2;
elseif fu>fpartition(3) && fu<fpartition(4)
        fdice = 3;
elseif fu>fpartition(4) && fu<fpartition(5)
        fdice = 4;
elseif fu>fpartition(5) && fu<fpartition(6)
        fdice = 5;
elseif fu>fpartition(6) && fu<fpartition(7)
        fdice = 6;
end

fprintf('result of roll of a fair dice is: %f \n',fdice)



%%  ferquency distribution of a simulated fair dice by barchart
    n = 1000;
    fairdice = 6*rand(1,n);
    fairdice = ceil(fairdice);
    figure;
    histogram(fairdice, 'Normalization', 'probability');
    title('Frequency distribution of simulated fair dice');
    samplemean = mean(fairdice);
    fprintf('Sample mean is %f \n', samplemean)
    samplevariance = var(fairdice);
    fprintf('Sample variance is %f \n', samplevariance); 

%% richiesta 3
% % % esercizio c
% % % checking the central limit theorem and weak law of great numebrs

figure
n=1000;
fairsamplevar=sum((vettoredado-3.5).^2)/6;
populmeanfair=repmat(3.5,1,n);
z_scoremean=repmat(0,1,n); 
 sample = nan(1,n);
 samplemeanfair = nan(1,n);
 assex = nan(1,n);
 zscore = nan(1,n);
 zmean = nan(1,n);
 for i = 1:n
    dado = randi(6);
    sample(i) = dado;
    assex(i) = i;
    samplemeanfair(i) = mean(sample, 'omitnan');
    zscore(i) = (dado-fairpopmean)/sqrt(fairsamplevar);
    zmean(i) = mean(zscore, 'omitnan');
    plot(assex, samplemeanfair, assex, zmean,assex ,populmeanfair ,assex ,z_scoremean)
    legend('samplemeanfair','zmean','populmeanfair=3.5','z-scoremean=0')
    drawnow
 end
% the theoretical intuition behind this property is the central limit theorem
%% to check the central limit theorem
% % % histogram of the distribution of the z-score

figure 
nsamples = 10000;
mean_samples = nan(1, nsamples);
samples_z_score=nan(1,nsamples);
sample=(0);
for i = 1:nsamples
    for k = 1:nsamples
        die = randi(6);
        sample(k) = die;
    end
    mean_samples = mean(sample,'omitnan');
    sd=sqrt(fairpopvar);
    samples_z_score(i)=(mean_samples-fairpopmean)/(sd/sqrt(nsamples));
end
varsampleszscore=var(samples_z_score);
histogram(samples_z_score,25,'normalization','pdf')
yyaxis right
x = -6:0.01:6;
y=normpdf(x,0,1);
plot(x,y)
%% fequency distribution of 10000 samples of a fair dice
figure
nsample=10000;
mean_nsample=nan(1,nsample);
dsample=(0);
for i=1:nsample
    for k=1:15
        dado=randi(6);
        dsample(k)=dado;
    end
    mean_nsample(i)=mean(dsample,'omitnan');
end
histogram(mean_nsample,25)
yyaxis right 
x=0:0.01:6;
y=1/(sqrt(2*pi)*2.9167)*exp(-(1/2).*(x-3.5).^2)/(2.9167);
plot(x,y)
title('Frequency distribution of 10000 samples of a fair dice')

