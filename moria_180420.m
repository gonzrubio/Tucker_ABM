clear ;clc;

rng('default') ;
s = rng ;

% function [peak_infection,days_to_peak_infection,...
%     peak_hospitalisation,days_to_peak_hospitalisation,...
%     peak_quarantine,days_to_peak_quarantine,...
%     total_proportion_infected,epidemic_over,track_states]=moria_new_qt

%CAMP STRUCTURE
Nb=8100;            %population in isoboxes
hb=810;             %number of isoboxes
iba=0.5;            %proportion of area covered by isoboxes
Nt=10600;            %people in tents
ht=10600/4;          %number of tents
N=Nb+Nt;            %total population
fblocks=[1,1];      %initial sectoring

%EMPIRICAL AGE AND SEX DISTRIBUTION
age_and_sex=readtable('age_and_sex.csv');       %read observed values from file
age_and_sex=age_and_sex{:,2:3};                 %convert to array and remove nuisnance column

%TRANSMISSION PARAMETERS
%Infection
twh=0.5;            %prob of infecting eah person in your household per day
aip=0.1;            %probability of infecting each person you meet, per meeting (Fang et al.)
tr=1;               %initial transmission reduction (relative to assumed per contact transmission rate, outside household only)


%OTHER PARAMETERS 
siprob=0;           %probability of spotting symptoms, per person per day
clearday=7;         %days in quarantine after no virus shedding (i.e., recovery)
pac=0.179;          %permanently asymptomatic cases (Mizumoto et al 2020 Eurosurveillance)
ss=0.20;            %realtive strength of interaction between different ethnicities
%Set initial movement parameters. Note that the initial assumption is that
%everyone uses the larger radius some proportion of the time, which is
%NOT the same as assuming that some people always use the larger radius,
%Nonetheless, I am setting the proportion equal to the number of males age
%10-50 in the population.
lr1=0.02;           %smaller movement radius
lr2=0.1;            %larger movement radius
lrtol=0.02;         %scale interactions - two people with completely overlapping rages with this radius interact once per day

%CREATE POPULATION (columns: home number, disease state, days to symptoms for this person,
%days passed in current state, whether this person will be asymptomatic, age, male, chronic, 
%wanderer), where wanderers use the larger radius
%Disease states are: 0 - susceptible, 1 - exposed, 2 - presymptomatic, 3 -
%symptomatic, 4 - mild, 5 - severe, 6 - recovered. Similar states in
%quarantine are the same plus 7.
%Assign each person to a houdehold. Start each person as susceptible.
rN=[ceil(hb*rand(Nb,1));hb+ceil(ht*rand(Nt,1))];
[~,~,ui]=unique(rN);
pop=[sort(ui),zeros(length(ui),1),];
%Assign one person to be exposed to the virus
pop(randsample(N,1),2)=1;
%Calculate days to first symptoms (if they develop symptoms) for each
%person, following (Backer et al. 2020 Eurosurveillance)
k=(2.3/6.4)^-1.086;
L=6.4/(gamma(1+1/k));
pop=[pop,round(wblrnd(L,k,N,1)),zeros(N,1),(rand(N,1)<pac*(N/(N-300)))];
%Assign age and sex following the observed distribution in the camp
pop=[pop,age_and_sex(randsample(size(age_and_sex,1),N,true),:)];
%Assign chronic states, matching the relationship between age and chronic
%states in the data
c_adj=fminsearch(@(adj)abs(300-sum((1+exp(-(adj-11.69+.2191*pop(:,6)-0.001461*pop(:,6).^2))).^-1)),4);
rchron=(1+exp(-(c_adj-11.69+.2191*pop(:,6)-0.001461*pop(:,6).^2))).^-1;
pop=[pop,rand(N,1)<rchron];
pop(pop(:,8)==1,5)=0;       %assue that people with chronic conditions are not asymptomatic, and correct number of asymptomatics above
pop=[pop,(pop(:,7)==1 & pop(:,6)>=10)];

%CREATE HOUSEHOLDS (pph holds people per household, and hhloc holds the x
%and y coordinates of each household)
%Get people per household
[hn,~,ci]=unique(pop(:,1));
pph=accumarray(ci,1);
%Count households (to save recalculating this later)
maxhh=length(pph);
%Assign household locations, with isoboxes occupying a square in the center
hhloc1=(1-sqrt(iba))/2+sqrt(iba)*rand(pop(Nb,1),2);
hhloc2=rand(pop(N)-pop(Nb),2);

hhloc2w=ceil(rand(pop(N)-pop(Nb),1)/.25);

hhloc2(hhloc2w==1,:)=[hhloc2(hhloc2w==1,1)*(1-sqrt(iba))/2+1-(1-sqrt(iba))/2,...
    hhloc2(hhloc2w==1,2)*(1-(1-sqrt(iba))/2)];

hhloc2(hhloc2w==2,:)=[hhloc2(hhloc2w==2,1)*(1-(1-sqrt(iba))/2)+(1-sqrt(iba))/2,...
    hhloc2(hhloc2w==2,2)*(1-sqrt(iba))/2+1-(1-sqrt(iba))/2];

hhloc2(hhloc2w==3,:)=[hhloc2(hhloc2w==3,1)*(1-sqrt(iba))/2,...
    hhloc2(hhloc2w==3,2)*(1-(1-sqrt(iba))/2)+(1-sqrt(iba))/2];

hhloc2(hhloc2w==4,:)=[hhloc2(hhloc2w==4,1)*(1-(1-sqrt(iba))/2),...
    hhloc2(hhloc2w==4,2)*(1-sqrt(iba))/2];

hhloc=[hhloc1;hhloc2];

%POSITION TOILETS on a grid evenly spaced throughout the camp
tblocks=[12,12];
tgroups=prod(tblocks);
tu=N/tgroups;
tgrid1=sum(hhloc(:,1)*ones(1,tblocks(1))>ones(maxhh,1)*(1:tblocks(1))/tblocks(1),2);
tgrid2=sum(hhloc(:,2)*ones(1,tblocks(2))>ones(maxhh,1)*(1:tblocks(2))/tblocks(2),2);
tnum=(tgrid2)*tblocks(1)+tgrid1+1;
%Create where 1 indicates that households share the same toilet
tshared=(tnum*ones(size(tnum))'==ones(size(tnum))*tnum')-eye(maxhh);

%CREATE ETHNIC GROUPS. Ethnic groups are spatailly clustered and randomly
%positioned. Households are assigned to ethnicities. The distribution
%matches the distribution of people in the Moria data. Only ethnicites
%with >50 people in the observed data included.
g=[7919,149,706,107,83,442,729];    %[Afghan,Cameroon,Congo,Iran,Iraq,Somalia,Syria]
g=round(maxhh*g/15327);             %number of household per group
g=datasample(g,length(g),'Replace',false);  %randomise group order
%Pick random household, and assign the n households nearest that location to
%an ethnicity
hhunass=[(1:size(hhloc,1))',hhloc];
hheth=zeros(maxhh,1);
for i=1:length(g)
    %Pick unassigned household as center
    gcen=hhunass(ceil(size(hhunass,1)*rand(1)),2:3);
    dfromc=sum((hhunass(:,2:3)-ones(size(hhunass,1),1)*gcen).^2,2);
    [~,cloind]=sort(dfromc);
    hheth(hhunass(cloind(1:g(i)),1))=i;
    hhunass(cloind(1:g(i)),:)=[];
end
%Scale local interaction strength for ethinicities
ethmatch=(hheth*ones(size(hheth))'==ones(size(hheth))*hheth');
ethcor=ethmatch+ss*(1-ethmatch);

%CREATE INTERACTION RATES. fshared: matrix that indictes whether two households 
%use the same food distribution center; lis: matrix that holds local
%interacton stregth between members of different households; tl: probability 
%of transmission per local interaction; tt: probability of transmission per
%regional interaction (i.e., 3x daily toilet visits, from either person
%before or person after), tg: probability of transmission per global
%interaction (i.e., 3x daily food lines, from either person before or 
%person after, but only 1/4 of people go to the food line each time to 
%collect for household)
[fshared,lis]=make_obs_180420(fblocks,lr1,lr2,N,hhloc,maxhh,lrtol,ethcor);

%LOOP TO SIMULATE INFECTION DYNAMICS
%Control variables
d2s=200;
i=0;
finished=0;
op=1;
%Tracking variables
track_states=zeros(d2s,14);
thosp=0;
while finished==0
    
    i=i+1;
    fprintf("Day #%i\n", i) ;
    
    %Record states
    track_states(i,:)=hist(pop(:,2),0:13);
    
    %Check to see if the epidemic has ended (and stop)
    if i==d2s
        finished=1;
    end
    if ((sum(track_states(i,2:6))+sum(track_states(i,8:14)))==0 & i>10)
        finished=1;
        fprintf("Epidemic ended on day #%i\n", i) ;
    end
    
    %Introduce new interventions on a cue
    if (op==1 & i>1 & (sum(cpih)-sum(cpco))/N>-1)
        fprintf("Introduced new interventions on a cue at iteration #%i\n", i) ;
        iat1=i ;
        op=0;
        lr1=0.001;
        fblocks=[12,12];
        tr=.25;
        [fshared,lis]=make_obs_180420(fblocks,lr1,lr2,N,hhloc,maxhh,lrtol,ethcor);
        %set rate for people to violate lockdown
        %assume people who violate do so at lr2
        viol_rate=0.05;
        pop(:,9)=(rand(N,1)<viol_rate);
    end
    
    %UPDATE DISEASE TRACKING
    %Add day since exposure or symptoms (if not unexposed)
    pop(:,4)=pop(:,4)+(pop(:,2)>0);
    
    %UPDATE IN POPULATION
    %Assign to recovered state
    mild_rec=(rand(N,1)>exp(log(.1)/5));        %Liu et al 2020 The Lancet
    sev_rec=(rand(N,1)>exp(log(63/153)/12));    %Cai et al. Preprint (note that "recovery" here could be death)
    pop(((pop(:,2)==4)+mild_rec)==2,2)=6;
    pop(((pop(:,2)==5)+sev_rec)==2,2)=6;
    %Assign people to cases severe enough to be hospitalised. These rates
    %are age speific. We will multiple Verity's rates by 1/(1-0.179) and
    %only allow symptomatic people to be hospitalised, to avoid changing
    %the proportion of asymptomatic infections in the population. We might
    %reasonably assume that only people with infections severe enough to be
    %hospitalised can move on to death, but I will not bother withthis for
    %now. Instead, I will simply calculate deaths from the total number of
    %people who were infected at the end of the simulation.
    %Get random numbers to determine health states
    pick_sick=rand(N,1);
    %First, everyone who was in state 3 progresses to at least state 4 
    pop(((pop(:,2)==3)+(pop(:,4)==6))==2,2)=4;
    %Now, they are assigned to severe cases with age-specific
    %probabilities as in Verity, and adjustments for pre-existing
    %conditions as in Tuite. This means that overall severe cases will be
    %slightly higher than Verity predicted if pre-existing conditions are
    %absorbed into their hospitalisation estimates. However, we cannot
    %correct for this without knowing the frequency of pre-existing
    %conditions in Tuite's data. Moreover, the problem may not be large.
    %Hospitalisation estimates are still lower than in Tuite's data.
    %Furthermore, at this point we lack any data at all on how severely
    %COVID-19 may affect populations that are initially malnourished or
    %otherwise in poor condition.
    asp=[0;.000408;.0104;.0343;.0425;.0816;.118;.166;.184];                 %Verity et al. hospitalisation
    aspc=[0.0101;0.0209;0.0410;0.0642;0.0721;0.2173;0.2483;0.6921;0.6987];  %Verity et al. corrected for Tuite
    for sci=0:7
        thosp=thosp+sum(((pop(:,2)==4)+(pop(:,4)==6)+(pick_sick<asp(sci+1))+(pop(:,6)>=10*sci)+(pop(:,6)<(10*sci+1)))==5);
        pop(((pop(:,2)==4)+(pop(:,4)==6)+(pick_sick<asp(sci+1))+(pop(:,6)>=10*sci)+(pop(:,6)<(10*sci+1)))==5,2)=5;
        thosp=thosp+sum(((pop(:,2)==4)+(pop(:,4)==6)+(pick_sick<aspc(sci+1))+(pop(:,6)>=10*sci)+(pop(:,6)<(10*sci+1))+(pop(:,8)==1))==6);
        pop(((pop(:,2)==4)+(pop(:,4)==6)+(pick_sick<aspc(sci+1))+(pop(:,6)>=10*sci)+(pop(:,6)<(10*sci+1))+(pop(:,8)==1))==6,2)=5;
    end
    thosp=thosp+sum(((pop(:,2)==4)+(pop(:,4)==6)+(pick_sick<asp(9))+(pop(:,6)>=80))==4);
    pop(((pop(:,2)==4)+(pop(:,4)==6)+(pick_sick<asp(9))+(pop(:,6)>=80))==5,2)=5;
    thosp=thosp+sum(((pop(:,2)==4)+(pop(:,4)==6)+(pick_sick<aspc(9))+(pop(:,6)>=80)+(pop(:,8)==1))==5);
    pop(((pop(:,2)==4)+(pop(:,4)==6)+(pick_sick<aspc(9))+(pop(:,6)>=80)+(pop(:,8)==1))==5,2)=5;
    %Move to presymptomatic to symptomatic but not yet severe
    pop(((pop(:,2)==2)+(pop(:,4)>=pop(:,3)))==2,[2,4])=ones(sum(((pop(:,2)==2)+(pop(:,4)>=pop(:,3)))==2),1)*[3,0];
    %Move to exposed to presymptomatic
    pop(((pop(:,2)==1)+(pop(:,4)>=floor(pop(:,3)/2)))==2,2)=2;
    
    %UPDATE IN QUARANTINE. See notes from population
    %Assign to recovered state (using mild_rec and sev_rev from above)
    pop(((pop(:,2)==11)+mild_rec)==2,[2,4])=ones(sum(((pop(:,2)==11)+mild_rec)==2),1)*[13,0];
    pop(((pop(:,2)==12)+sev_rec)==2,[2,4])=ones(sum(((pop(:,2)==12)+sev_rec)==2),1)*[13,0];
    %Assign people to cases severe enough to be hospitalised (using
    %pick_sick from above)
    %First, everyone who was in state 10 progresses to at least state 11 
    pop(((pop(:,2)==10)+(pop(:,4)==6))==2,2)=11;
    %Now, they are assigned to severe cases 
    for sci=0:7
        thosp=thosp+sum(((pop(:,2)==11)+(pop(:,4)==6)+(pick_sick<asp(sci+1))+(pop(:,6)>=10*sci)+(pop(:,6)<(10*sci+1)))==5);
        pop(((pop(:,2)==11)+(pop(:,4)==6)+(pick_sick<asp(sci+1))+(pop(:,6)>=10*sci)+(pop(:,6)<(10*sci+1)))==5,2)=12;
        thosp=thosp+sum(((pop(:,2)==11)+(pop(:,4)==6)+(pick_sick<aspc(sci+1))+(pop(:,6)>=10*sci)+(pop(:,6)<(10*sci+1))+(pop(:,8)==1))==6);
        pop(((pop(:,2)==11)+(pop(:,4)==6)+(pick_sick<aspc(sci+1))+(pop(:,6)>=10*sci)+(pop(:,6)<(10*sci+1))+(pop(:,8)==1))==6,2)=12;
    end
    thosp=thosp+sum(((pop(:,2)==11)+(pop(:,4)==6)+(pick_sick<asp(9))+(pop(:,6)>=80))==4);
    pop(((pop(:,2)==11)+(pop(:,4)==6)+(pick_sick<asp(9))+(pop(:,6)>=80))==5,2)=12;
    thosp=thosp+sum(((pop(:,2)==11)+(pop(:,4)==6)+(pick_sick<aspc(9))+(pop(:,6)>=80)+(pop(:,8)==1))==5);
    pop(((pop(:,2)==11)+(pop(:,4)==6)+(pick_sick<aspc(9))+(pop(:,6)>=80)+(pop(:,8)==1))==5,2)=12;
    %Move to presymptomatic to symptomatic but not yet severe
    pop(((pop(:,2)==9)+(pop(:,4)>=pop(:,3)))==2,[2,4])=ones(sum(((pop(:,2)==9)+(pop(:,4)>=pop(:,3)))==2),1)*[10,0];
    %Move to exposed to presymptomatic
    pop(((pop(:,2)==8)+(pop(:,4)>=floor(pop(:,3)/2)))==2,2)=9;
    
%     if sum(pop(:,2)==2)>0
%         return
%     end

    %IDENTIFY CONTAGIOUS AND ACTVE PEOPLE IN DIFFERENT CONTEXTS
    %Contagious in the house and at toilets, in population
    cpih=(pop(:,2)>1 & pop(:,2)<6);  
    %Contagious in the house, in quarantine
    cpihq=(pop(:,2)>8 & pop(:,2)<13);  
    %Contagious at large in the population (all - for food lines)
    cpco=(pop(:,2)==2 | (pop(:,2)>2 & pop(:,2)<5 & (pop(:,5)==1 | pop(:,6)<16)));
    %All at large in the population (all - for food lines)
    apco=(cpco==1 | pop(:,2)<2 | pop(:,2)==6);
    %Contagious at large in the population (sedentaries)
    cpcos=(cpco==1 & pop(:,9)==0);
    %Contagious at large in the population (wanderers)
    cpcow=(cpco==1 & pop(:,9)==1);
    
    %COMPUTE INFEECTED PEOPLE PER HOUSEHOLD
    infh=accumarray(ci,cpih);   %all infected in house and at toilets, population 
    infhq=accumarray(ci,cpihq); %all infected in house, quarantine 
    infl=accumarray(ci,cpco);   %presymptomatic and asymptomatic for food lines
    allfl=accumarray(ci,apco);  %all people in food lines
    infls=accumarray(ci,cpcos); %all sedentaries for local transmission
    inflw=accumarray(ci,cpcow); %all wanderers for local transmission
    
    %COMPUTE INFECTION PROBABILITIES FOR EACH PERSON BY HOUSEHOLD
    %Probability for members of each household to contract from their housemates
    cfh=1-(1-twh).^infh;        %in population
    cfhq=1-(1-twh).^infhq;      %in quarantine
%CHECK THIS MID-RUN!!!! 
    %compute proportions infecteds at toilets and in food lines
    pitoil=(tshared*infh)./(tshared*pph);
    pifl=(fshared*infl)./(fshared*allfl);
    %compute transmission at toilets by household
    xi=0:6;
    trans_at_toil=1-factorial(6)*sum(((factorial(6-xi).*factorial(xi)).^-1).*...
        ((1-pitoil).^(6-xi)).*(pitoil.^xi).*(1-aip*tr).^xi,2);
    %compute transmission in food lines by household. Assume each person
    %goes to the food line once per day on 75% of days. Other days someone
    %brings food to them (with no additional contact)
    xi=0:2;
    trans_in_fl=(3/4)*(1-factorial(2)*sum(((factorial(2-xi).*factorial(xi)).^-1).*...
        ((1-pifl).^(2-xi)).*(pifl.^xi).*(1-aip*tr).^xi,2));
    %household in quarantine don't get these exposures, but that is taken
    %care of below because this is applied only to susceptibles in the
    %population with these, we can calculate the probability of all 
    %transmissions that calculated at the household level
    pthl=1-(1-cfh).*(1-trans_at_toil).*(1-trans_in_fl);
    %transmissions during movement around the residence must be calculated
    %at the individual level, because they depend not on what movement
    %radius the individual uses. So...

    %Calculate expected contacts with infected for individuals that use
    %small and large movement radii
    lr1_exp_contacts=lis(:,:,1)*infls+lis(:,:,2)*inflw;
    lr2_exp_contacts=lis(:,:,2)*infls+lis(:,:,3)*inflw;
    %but contacts are roughly Poisson distributed (assuming a large
    %population), so the transmission rates are
    trans_for_lr1=1-exp(-lr1_exp_contacts*aip*tr);
    trans_for_lr2=1-exp(-lr2_exp_contacts*aip*tr);
    %now, assign the appropriate local transmission rates to each person
    local_trans=trans_for_lr1(pop(:,1)).*(1-pop(:,9))+trans_for_lr2(pop(:,1)).*pop(:,9);
    
    %Finally, compute the full per-person infection probability within
    %households, at toilets and food lines, 
    full_inf_prob=1-(1-pthl(pop(:,1))).*(1-local_trans);
        
    %ASSIGN NEW INFECTIONS
    %Find new infections by person, population
    new_inf=full_inf_prob>rand(N,1);
    %Impose infections, population
    pop(:,2)=pop(:,2)+(1-sign(pop(:,2))).*new_inf;
    %Find new infections by person, quarantine
    new_inf=cfhq(pop(:,1))>rand(N,1);
    %Impose infections, quarantine
    pop(:,2)=pop(:,2)+(pop(:,2)==7).*new_inf;
    
    %MOVE HOUSEHOLDS TO QUARANTINE
    %Identify symptomatic people in population
    sip=(pop(:,2)>2 & pop(:,2)<6 & pop(:,5)==0 & pop(:,6)>=16).*(rand(N,1)<siprob);
    %Identify symptomatic households
    symphouse=unique(pop(sip==1,1));
    %Move households to quarantine
    pop(ismember(pop(:,1),symphouse),2)=pop(ismember(pop(:,1),symphouse),2)+7;
    
    %MOVE INDIVIDUALS BACK TO POPULATION when they have not shed for
    %cleardays. Currently, the model does not include a mechanism for
    %sending people back from quarantine if they never get the infection.
    %This may not matter if the infection spreads in quarantine, but we
    %will have to watch to see if some people get stuck in state 7.
    pop((pop(:,2)==13 & pop(:,4)>=7),2)=6;
    
    %TRACK INFECTION THROUGH SPACE
%     if rem(i,5)==0
%         figure(1)
%         scatter(hhloc(:,1),hhloc(:,2),(1+infh).^2,hheth)
%         drawnow
%     end
    
end

% %TRACK INFECTION THROUGH TIME
% figure(2)
% plot(track_states(1:i,1:7))
% legend('susc','exp','asy','sym','mrec','sev','rec')
% % 
% figure(3)
% plot(track_states(1:i,2:6))
% legend('exp','asy','sym','mrec','sev')
% % 
% figure(4)
% plot(track_states(1:i,8:14))
% legend('susc','exp','asy','sym','mrec','sev','rec')
% 
% figure(5)
% plot(sum(track_states(1:i,8:14),2))
% legend('total quarantine')
% 
% figure(6)
% plot(sum(track_states(1:i,1:7),2))
% legend('total camp')

%COLLECT USEFUL DATA
[peak_infection,days_to_peak_infection]=max(sum(track_states(:,[2:6,9:13]),2));
[peak_hospitalisation,days_to_peak_hospitalisation]=max(sum(track_states(:,[6,13]),2));
[peak_quarantine,days_to_peak_quarantine]=max(sum(track_states(:,8:14),2));
total_proportion_infected=track_states(i,7)/N;
epidemic_over=(sum(track_states(i,[1,7]))==N);

%The old model had apparatus for calculating age and sex-specific death
%rates. I have not moved that to this model, for two reasons. First, unless
%the model has some behaviour by which the infection rate differs among age
%groups, and this model doesn't, then the total number of infected people
%and the total hospitalisations tell us everything we need to know about
%deaths. That is, if we reduce infections by x%, we will reduce deaths by
%x%, unless swamping the hospitals affects the death rate, in which case
%the number of concurrent hospitalisations tell us what we need to know
%Second, we really don't know anything about what kinds of death rates to
%expect in an undernourished, impoverished poplation with inadequate
%medical facilities. Prelilminary data from US minority populations
%suggests that these things matter. So, anything we do with death rates at
% this point is really just making stuff up. Above I have turned ICU and
% death rates to -1 because other functions want them, but we can ignore
% them for now.

