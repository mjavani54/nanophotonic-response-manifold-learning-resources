%################################################
% DATASET GENERATION FOR SCATTERING FROM MULTI-LAYERED METAMATERIALS
%###############################################
% This program calculates and plots refraction and transmission coefficients
% of multi-layered structures as a function of frequency for a specific
% angle of incidence. The structures and their responses are then stored in
% .csv files.
%
% Code adapted from Umit Cotuk's Thesis found here
% (http://hdl.handle.net/10945/2115)
%
% Author: Reid Barton, 9/14/2020

clear;
clc;

%% Some helpful material properties defined
epsilon_Sio2=1.46;
epsilon_SiN3=3.1;
epsilon_Tio2=5.913;    % was 5.913;


%% Define Dataset parameters
fileName = 'SiO2_TiO2_8_layer_50K_200_points';      % Dataset file name

NDataPoints = 50000;             % Number of structures to generate
NSamp = 200;                    % Number of samples in the response

fmin = 250e3;                   % min frequency in GHz (ex. 450e3 -> 450 THz)
fmax = 1050e3;                   % max frequency in GHz
f=linspace(fmin,fmax, NSamp);   % frequency range

p = 1;                          % p = Polarization Type Parallel (1), Perpendicular (2)
th = 0;                         % th = Incidence Angle (Degree)

%% Define structure properties
N = 8;                          % num layers

lmin = 30e-9;                  % Min layer height (m)
lmax = 70e-9;                  % Max layer height (m)

epsilon_1 = epsilon_Sio2;       % first material Permittivity(€)
epsilon_2 = epsilon_Tio2;       % second material Permittivity(€)

Mu_1 = 1;                       % first material Permeability(μ)
Mu_2 = 1;                       % second material Permeability(μ)

sigma_1 = 0;                    % first material Loss (sigma)
sigma_2 = 0;                    % second materail Loss (sigma)


%% Create Data Set
tic;
h = waitbar(0,'','CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
setappdata(h,'canceling',0);

responses = zeros(NSamp, NDataPoints)';
device_layers = zeros(N, NDataPoints)';

for device = 1:NDataPoints
    % Create one device structure
    layers = rand(N,1).*(lmax-lmin) + lmin;
    for layer=1:N
        l(layer) = layers(layer);
        if mod(layer,2)==1
            Mu(layer) = Mu_1;
            Ei(layer) = epsilon_1;
            sigma(layer) = sigma_1;
        else
            Mu(layer) = Mu_2;
            Ei(layer) = epsilon_2;
            sigma(layer) = sigma_2;
        end
    end

    response = calculate_device(f, th, p, Mu, Ei, l, sigma);
    responses(device, :) = response;
    device_layers(device, :) = layers;
    waitbar(device/NDataPoints, h,'Generating Data');

    if getappdata(h,'canceling')
        break
    end

end
delete(h)

dlmwrite([fileName '_RS.csv'], responses', '-append')
dlmwrite([fileName '_DS.csv'], device_layers', '-append')

toc;

%%

function response = calculate_device(f, th, p, Mu, Ei, l, sigma)
% f = Frequency Range (Ghz)
% th = Incidence Angle (Degree)
% p = Polarization Type Parallel (1), Perpendicular (2)
% Mu, Ei, l, sigma = properties for each layer

N = size(Mu, 2);

%Constants
E0=8.85e-12; %Permittivity of free space
Mu0=12.5664e-7; %Permeability of free space
c=1/sqrt(E0*Mu0); %Speed of light (m/s)

thr=th*pi/180; %Angle of incidence (Radian)
st=sin(thr);
ct=cos(thr);
for i=1:N
    if Ei(i)<0
        E(i)=abs(Ei(i));
    else
        E(i)=Ei(i);
    end
end
%This part calculates the total reflection and transmission coefficients of
%the structure
for k =1:length(f);
    lambda=c/(f(k)*1e9);
    Beta1=2*(pi/lambda);
    total=1;
    w(k)=2*pi*f(k)*1e9;
    %Calculating the characteristic impedance of the layers
    for i=1:N
        Er(i)=E(i)-(j*sigma(i)/w(k)/E0);
        if p==1
            Z(i)=sqrt(Er(i)*Mu(i)-st^2)/Er(i)/ct;
        elseif p==2
            Z(i)=Mu(i)*ct/sqrt(Er(i)*Mu(i)-st^2);
        else
            display ('Polarization type must be 1 or 2');
        end
    end
    %Individual reflection and transmission coefficients of the layers
    ro(1)=(Z(1)-1)/(Z(1)+1);
    tao(1)=ro(1)+1;
    for i=2:N
        ro(i)=(Z(i)-Z(i-1))/(Z(i)+Z(i-1));
        tao(i)=ro(i)+1;
    end
    %Generating wave matrices
    for i =1:(N-1)
        if Ei(i)<0
            sign=-1;
        else
            sign=1;

        end
        Beta=Beta1*sign;
        A=exp(j*Beta*l(i)*sqrt(Er(i)-st^2));
        Ac=exp(-j*Beta*l(i)*sqrt(Er(i)-st^2));
        total=total*((1/tao(i))*[A,ro(i)*Ac;ro(i)*A,Ac]);
    end
    %Total reflection and transmission coefficients of the structure
    total=1/tao(N)*total*[1 ro(N); ro(N) 1];
    rot(k)=total(2,1)/total(1,1);
    taot(k)=1/total(1,1);
end
%This part plots the coefficients as a function of frequency
% plot(f,abs(rot),'r-',f,abs(taot),'--');
% grid;
% xlabel('Frequency (GHz)');
% ylabel('Coefficient Magnitude');
% legend('Reflection Coefficient','Transmission Coefficient');

response = abs(taot);
end






