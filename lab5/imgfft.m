clc
clear
close all
imapath='D:\graduated\Image_process\lab\PGM_images\crosses.pgm';
ima=imread(imapath);
ima=double(ima);
fima=fft2(ima);
shifima=fftshift(fima);
x=1:size(ima,1);
y=1:size(ima,2);
[Y,X]=meshgrid(x,y);
XY=(-1).^(X+Y);
%myshifima=fft2(ima.*XY);
myfima= mydft2(ima.*XY);
myifima= myidft2(myfima);
myshifima=myifima.*XY;

mag=abs(shifima);
theta=getangle(shifima);
an=angle(shifima);

rima=ifft2(shifima);
myrima=ifft2(myshifima).*XY;
rmag=ifft2(mag);
rphase=ifft2(theta);
ran=ifft2(an);




figure(1)
subplot(221)
imshow(disima(ima))
title('original')
subplot(222)
imshow(disima(log(fima)))
title('no shift FFT')
subplot(223)
imshow(disima(log(shifima)))
title('shiftFFT, save')
subplot(224)
imshow(disima(log(myshifima)))
title('myshift -1(x,y)')

figure(2)
subplot(221)
imshow(disima(log(mag)))
title('自带幅度')
subplot(222)
imshow(disima(an))
title('自带相位')
subplot(223)
imshow(disima(theta))
title('my theta')
subplot(224)
imshow(disima(myrima))
title('ifft -1 xy')

figure(3)
subplot(221)
imshow(disima(log(rmag)))
title('自带幅度反')
subplot(222)
imshow(disima(abs(ran)))
title('自带相位反')
subplot(223)
imshow(disima(rima))
title('ifft反')
subplot(224)
imshow(disima(abs(rphase)))
title('theta相位反')

figure(4)
subplot(221)
imshow(disima(log(myfima)))
title('mydft2 shift')
subplot(222)
imshow(disima(myifima))
title('myifima')
subplot(223)
imshow(disima(myshifima))
title('myshifima')
subplot(224)
imshow(disima(abs(rphase)))
title('theta相位反')


function ang=getangle(Z)
ang=zeros(size(Z));
for i=1:size(Z,1)
    for j=1:size(Z,2)
        z=Z(i,j);
        R=real(z);
        I=imag(z);
        if R==0 
            if I>0
                an=pi/2;
            elseif I==0
                an=0;
            else
                an=-pi/2;
            end
        else
            an=atan(I/R);
            if R<0&&I>0
                an=an+pi;
            elseif R<0&&I<0
                an=an-pi;
            end
            
            
        end
        ang(i,j)=an;
    end
end
end
            
            
            


function dima= disima(ima)

ima=abs(ima);
ima(ima==inf)=0;
maxL=255;
minL=0;
mL=maxL-minL;
maxv=max(max(ima));
minv=min(min(ima));
L=maxv-minv;
dima=(ima-minv)*mL/L+minL;
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
            temp=temp+I(x+1,y+1)*exp(-1i*2*pi*(u*x/M+v*y/N));
            end
        end
        
        myfima(u+1,v+1)=myfima(u+1,v+1)+temp;
       
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
            temp=temp+I(u+1,v+1)*exp(1i*2*pi*(u*x/M+v*y/N));
            end
        end
        myfima(x+1,y+1)=real(temp);

    end
end

end