
close all
clear
clc
%%
imapath='D:\graduated\Image_process\lab\PGM_images\';
savepath='D:\graduated\Image_process\lab\lab_report\lab6\matlab2\';
namelist={ 'bridge', 'goldhill'};
Doset=[ 60 80 120 160 200 240 ];
GL=[  0.1 0.25 0.5 ];
GH=[ 1.5 2 3 ];
%% homomorphic filter 
for i=1:length(namelist)
    name=namelist{i};
    imgpath=[imapath,name,'.pgm'];
    ima=imread(imgpath);
    ima=ima(:,:,1);
    ima=double(ima)+1;
    imwrite(disima(ima),[savepath,name,'_original','.jpg'])
     fimg=fftshift(fft2(ima));
     imwrite(disima(log(real(fimg))),[savepath,name,'_F','.jpg']);
     
     lnima=log(ima);
     fimg=fftshift(fft2(lnima));
     imwrite(disima(log(real(fimg))),[savepath,name,'_EF','.jpg']);
     %fimg= mydft2(ima);
    for k=1:length(Doset)
         for j=1:length(GL)
              for m=1:length(GH)
            Do=Doset(k);
            gl=GL(j);
            gh=GH(m);
            H=homo(gl,gh,Do,1,lnima);
            mark=['_homo_gl_',num2str(gl*100),'_gh_',num2str(gh*100),'_D0_',num2str(Do)];
            iimg=HOMOF(H,fimg,mark,name);


              end
         end
    end

end
%%  sinusoidal noise
namelist={ 'lena'};

for i=1:length(namelist)
    name=namelist{i};
    imgpath=[imapath,name,'.pgm'];
    ima=imread(imgpath);
    ima=ima(:,:,1);
    ima=double(ima);
    imwrite(disima(ima),[savepath,name,'_original','.jpg'])
    fimg=fftshift(fft2(ima));
    imwrite(disima(fimg),[savepath,name,'_F','.jpg'])
    
    Nima=addSiNoise(ima);
    imwrite(disima(Nima),[savepath,name,'_Noise','.jpg'])
    Nfimg=fftshift(fft2(Nima));
    imwrite(disima(Nfimg),[savepath,name,'_Noise_F','.jpg'])
    
        Do=100;
        w=10;
        
        H=IBJF(Do,w,Nima);
        imwrite(disima(H),[savepath,name,'_IBJHF','.jpg'])
        mark=['_IBJF_D0_',num2str(Do),'_w_',num2str(w)];
        img=BJ(H,Nfimg,savepath,name,mark);
        
        H=BBJF(Do,w,2,Nima);
        imwrite(disima(H),[savepath,name,'_BBJHF','.jpg'])
        mark=['_BBJF_D0_',num2str(Do),'_w_',num2str(w)];
        img=BJ(H,Nfimg,savepath,name,mark);
        
        H=GBJF(Do,w,Nima);
        imwrite(disima(H),[savepath,name,'_GBJHF','.jpg'])
        mark=['_GBJF_D0_',num2str(Do),'_w_',num2str(w)];
        img=BJ(H,Nfimg,savepath,name,mark);
        

end
%%  corelaction
imgpath=[savepath,'face','.jpg'];
face=double(rgb2gray(imread(imgpath)));
imgpath=[savepath,'face_back','.jpg'];
face_back=double(rgb2gray(imread(imgpath)));
imgpath=[savepath,'back','.jpg'];
back=double(rgb2gray(imread(imgpath)));
[pface,pback]=padding(face,back);
imwrite(disima(pface),[savepath,'face','_padding','.jpg'])
imwrite(disima(pback),[savepath,'back','_padding','.jpg'])
conv_F_B=ifft2(conj(fft2(pface)).*fft2(pback));
imwrite(disima(conv_F_B),[savepath,'conv','_face_back','.jpg']);
Oconv_F_B=removePadd(conv_F_B,back);
imwrite(disima(Oconv_F_B),[savepath,'conv_RP','_face_back','.jpg']);

[pface,pface_back]=padding(face,face_back);
imwrite(disima(pface_back),[savepath,'face_back','_padding','.jpg']);
conv_F_FB=ifft2(conj(fft2(pface)).*fft2(pface_back));
%imwrite(disima(conv_F_FB),[savepath,'conv','_face_face_back','.jpg']);
Oconv_F_B=removePadd(conv_F_FB,face_back);
imwrite(disima(Oconv_F_B),[savepath,'conv_RP','_face_face_back','.jpg']);












%%

%% padding
function [pima1,pima2]=padding(ima1,ima2)
M1=size(ima1,1);
N1=size(ima1,2);
M2=size(ima2,1);
N2=size(ima2,2);
temp=zeros(M1+M2-1,N1+N2-1);
temp(1:M1,1:N1)=ima1;
pima1=temp;

temp=zeros(M1+M2-1,N1+N2-1);
temp(1:M2,1:N2)=ima2;
pima2=temp;
end

%% padding
function ima1=removePadd(pima1,oima1)
M1=size(oima1,1);
N1=size(oima1,2);


ima1=pima1(1:M1,1:N1);


end
%%


function H=homo(gl,gh,Do,c,lnima)

 N=size(lnima,1);
    M=size(lnima,2);
    H=ones(M);
    for x=1:M
        for y=1:N
            u = x - M / 2;
			v = y-N / 2;
            R=sqrt(u^2+v^2);
       
                H(y,x)=(gh-gl)*(1-exp(-1*c*(R/Do)^2))+gl;

        end
    end

end


function ima=HOMOF(H,fimg,mark,name)

savepath='D:\graduated\Image_process\lab\lab_report\lab6\matlab2\';

F=fimg.*H;
%imwrite(disima(log(F)),[savepath,name,mark,cot,'_homoF_log','.jpg'])
ifimg=ifft2(fftshift(F)); 
%ifimg=myidft2(F);
ima=exp(real(ifimg));
imwrite(disima(ima),[savepath,name,mark,'.jpg'])
end
%%
function Nimg=addSiNoise(ima)

N=size(ima,1);
M=size(ima,2);
noise=zeros(N,M);
for x=1:M
    for y=1:N
       
        r=100;
        ux=60;
        vy=80;
       
        noise(y,x)=255*(cos(2*pi*r*(x-1)/M)+cos(2*pi*r*(y-1)/N)+cos(2*pi*(ux*(x-1)/M+vy*(y-1)/N))+cos(2*pi*(-1*ux*(x-1)/M+vy*(y-1)/N)));%sin(2*pi*(ux*(x-1)/M+vy*(y-1)/N))

    end
end
Nimg=ima+noise;

end





function img=BJ(H,Nfimg,savepath,name,mark)
f=H.*Nfimg;
imwrite(disima(f),[savepath,name,mark,'_F','.jpg'])
img=ifft2(fftshift(f));
imwrite(disima(img),[savepath,name,mark,'.jpg'])




end

function H=IBJF(Do,w,ima)
    M=size(ima,2);
    N=size(ima,1);
    H=ones(M);
    r=Do;
    for x=1:M
        for y=1:N
            u = x - M / 2;
			v = y - N / 2;
            R=sqrt(u^2+v^2);
            if abs(R-r)<=w/2
                H(y,x)=0;
            end

        end
    end

end

function H=BBJF(Do,w,n,ima)
    M=size(ima,2);
    N=size(ima,1);
    H=ones(M);
    r=Do;
    for x=1:M
        for y=1:N
            u = x - M / 2;
			v = y - N / 2;
            R=sqrt(u^2+v^2);
            
                H(y,x)=1/(1+(R*w/(R^2-Do^2))^(2*n));
         

        end
    end

end


function H=GBJF(Do,w,ima)
     M=size(ima,2);
    N=size(ima,1);
    H=ones(M);
    r=Do;
    for x=1:M
        for y=1:N
            u = x - M / 2;
			v = y - N / 2;
            R=sqrt(u^2+v^2);
            th=(R^2-Do^2)/(R*w);
            
                H(y,x)=1-1/exp(th^2);
         

        end
    end

end





function dima= disima(oima)

ima=round(abs(oima));
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



