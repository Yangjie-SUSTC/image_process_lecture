
close all
clear
clc
%%
imapath='D:\graduated\Image_process\lab\PGM_images\';
savepath='D:\graduated\Image_process\lab\lab_report\lab5\matlab2\';
namelist={ 'bridge', 'lena'};
Doset=[ 0.1, 0.3, 0.6, 0.9 ];
for i=1:length(namelist)
    name=namelist{i};
    imgpath=[imapath,name,'.pgm'];
    ima=imread(imgpath);
    ima=ima(:,:,1);
    ima=double(ima);
    imwrite(disima(ima),[savepath,name,'_original','.jpg'])
    
     fimg=fftshift(fft2(ima));
     %fimg= mydft2(ima);
    for k=1:length(Doset)
        Do=Doset(k);
        HIL=ILPF(Do,ima);
        HBL=BLPF(Do,2,ima);
        HGL=GLPF(Do,ima);
       
        ifimg=FF(HIL,fimg,'_ILPF',Do,name);
        ifimg=FF(HBL,fimg,'_BLPF',Do,name);
        ifimg=FF(HBL,fimg,'_GLPF',Do,name);
    end

end

namelist={ 'fingerprint1', 'fingerprint2'};

for i=1:length(namelist)
    name=namelist{i};
    imgpath=[imapath,name,'.pgm'];
    ima=imread(imgpath);
    ima=ima(:,:,1);
    ima=double(ima);
    imwrite(disima(ima),[savepath,name,'_original','.jpg'])
    
    fimg=fftshift(fft2(ima));
    %fimg= mydft2(ima);
    for k=1:length(Doset)
        Do=Doset(k);
        HPFI(fimg,name,Do,ima);
        HPFB(fimg,name,Do,ima);
        HPFG(fimg,name,Do,ima);
        
    end
  
   

end



function ifimg=FF(H,fimg,mark,Do,name)
cot=['_',num2str(Do*10)];
savepath='D:\graduated\Image_process\lab\lab_report\lab5\matlab2\';
imwrite(disima(log(H*1000000000+1)),[savepath,name,mark,cot,'_H','.jpg'])
imwrite(disima(log(fimg)),[savepath,name,mark,cot,'_F_log','.jpg'])
F=fimg.*H;
imwrite(disima(log(F)),[savepath,name,mark,cot,'_HF_log','.jpg'])
ifimg=ifft2(F);
%ifimg=myidft2(F);
imwrite(disima(real(ifimg)),[savepath,name,mark,cot,'.jpg'])
end

function []=HPsharp(ima,H,fimg,mark,Do,name)
cot=['_',num2str(Do*10)];
savepath='D:\graduated\Image_process\lab\lab_report\lab5\matlab2\';
ifimg=FF(H,fimg,mark,Do,name);

sima=real(ifimg);%


sharpima=im2bw(disima(sima)) ;
imwrite(disima(sharpima),[savepath,name,mark,'_sharp',cot,'.jpg'])



end

function H=ILPF(Do,ima)
    M=size(ima,1);
    N=size(ima,2);
    H=ones(M);
    r=Do*M/2;
    for x=1:M
        for y=1:N
            u = x - M / 2;
			v = N / 2 - y;
            R=sqrt(u^2+v^2);
            if R>r
                H(x,y)=0;
            end
                
            
        end
    end

end

function H=BLPF(Do,n,ima)
    M=size(ima,1);
    N=size(ima,2);
    H=ones(M);
    r=Do*M/2;
    for x=1:M
        for y=1:N
            u = x - M / 2;
			v = N / 2 - y;
            R=sqrt(u^2+v^2);
        
            H(x,y)=1/(1+(R/r)^(2*n));
   
        end
    end

end


function H=GLPF(Do,ima)
    M=size(ima,1);
    N=size(ima,2);
    H=ones(M);
    r=Do*M/2;
    for x=1:M
        for y=1:N
            u = x - M / 2;
			v = N / 2 - y;
            R=sqrt(u^2+v^2);
            a=0.5*(R/r)^2;
            H(x,y)=1/exp(a);
   
        end
    end

end




function H=HPFG(fimg,name,Do,ima)
H=-1*GLPF(Do,ima);
H=H-min(min(H));
H=1-GLPF(Do,ima);
HPsharp(ima,H,fimg,'_HPFG',Do,name);
end

function H=HPFI(fimg,name,Do,ima)
H=-1*ILPF(Do,ima);
H=H-min(min(H));
H=1-ILPF(Do,ima);
HPsharp(ima,H,fimg,'_HPFI',Do,name);
end

function H=HPFB(fimg,name,Do,ima)
H=-1*BLPF(Do,4,ima);
H=H-min(min(H));
H=1-BLPF(Do,4,ima);
HPsharp(ima,H,fimg,'_HPFB',Do,name);
end

function dima= disima(ima)

ima=round(abs(ima));
ima(ima==inf)=0;
maxL=255;
minL=0;
mL=maxL-minL;
maxv=max(max(ima));
minv=min(min(ima));
L=maxv-minv;
dima=(ima-minv)*mL/L+minL;
%dima=histeq(dima);
dima=uint8(dima);

end




function myfima= mydft2(I)

M=size(I,1);
N=size(I,2);
myfima=zeros(M,N);
for u=0:M-1
    for v=0:N-1
        temp=0;
        for x=0:M-1
            for y=0:N-1
            temp=temp+I(x+1,y+1)*((-1)^(x+y))*exp(-1i*2*pi*(u*x/M+v*y/N));%*((-1)^(x+y))
            end
        end
        
        myfima(u+1,v+1)=temp;
       
    end
end

end

function myfima= myidft2(I)

M=size(I,1);
N=size(I,2);
myfima=zeros(M,N);
for x=0:M-1
    for y=0:N-1
        temp=0;
        for u=0:M-1
            for v=0:N-1
                temp=temp+I(u+1,v+1)*exp(1i*2*pi*(u*x/M+v*y/N));%*((-1)^(x+y))
              
            end
        end
        myfima(x+1,y+1)=real(temp)/(M*N)*((-1)^(x+y));

    end
end

end