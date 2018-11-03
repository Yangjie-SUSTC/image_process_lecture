
close all
clc
clear
%%
function img=addSiNoise()
noise=zeros(100,80);
N=size(noise,1);
M=size(noise,2);
for x=1:M
    for y=1:N
        lx= round(M/2);
        ly=round(N/2);
        r=10;
        ux=3*r;
        vy=2*r;
        us=ux+lx;% fftshiftºóÍ¼Ïñ×ø±ê
        vs=ly+vy; 
        u=mod((us+lx)-1,M)+1; %fft×ø±ê
        v=mod((vs+ly)-1,N)+1;
        
        theta=u*(x-1)/M+v*(y-1)/N;
        noise(y,x)=exp(1j*2*pi*theta)/(M*N);
        th=ux*(x-1)/M+vy*(y-1)/N;
        img(y,x)=sin(2*pi*th);
       
      
        %noise(x,y)=a+b;
        
    end
end
end

fima=fftshift(fft2(noise));
ff=fftshift(fft2(img));


figure()
subplot(221)
imshow(noise,[])
subplot(222)
imshow(log(fima),[])
subplot(223)
imshow(img,[])
subplot(224)
imshow(log(ff),[])


