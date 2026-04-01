% ---------------------------------------------------------
% Gamma evaluation 2D: simple algorithm
% ---------------------------------------------------------
% P Dvorak 2018a
% ---------------------------------------------------------
% Quantitative comparison of two 2D dose distributions.
% Same fixed resolution (1x1 mm2) aligned rectangular dose matrices, normalized.
% No interpolation. Global normalization of dose difference.
% ---------------------------------------------------------
%
% INPUT:
% - doseEvaluated: evaluated 2D dose distribution, rectangular matrix,
%                  resolution 1x1 mm2
% - doseReference: reference 2D dose distribution, same size matched matrix as doseEvaluated,
%                  resolution 1x1 mm2
% -         dDose: dose difference tolerance in [%]
% -     dDistance: distance to agreement tolerance in [mm]
% -     Threshold: threshold level for gamma calculation in [%] and doseEvaluated
% OUTPUT:
% -   gammaMatrix: 2D matrix of gamma values, same size and resolution as
%                  evaluated dose distribution
% ---------------------------------------------------------
%
% Syntax: myGammaSimple(doseEvaluated,doseReference,dDose,dDistance,Treshold)
% ---------------------------------------------------------
%
% Dependent user-functions:
% - gammaColormap
% ---------------------------------------------------------
%
%         Version: 1/2018
% Version History: none
% ---------------------------------------------------------

function [pass_rate, gammaMatrix] = myGammaGlobal(doseEvaluated,doseReference,dDose,dDistance,Threshold)

% timeStart=tic;
% doseEvaluated = doseEvaluated*100/max(max(doseReference));
% doseReference = doseReference*100/max(max(doseReference));

% doseEvaluated = doseEvaluated*100/doseReference(151,95);
% doseReference = doseReference*100/doseReference(151,95);
gammaMatrix=-ones(size(doseEvaluated));

for i=1:size(doseEvaluated,1)
    for j=1:size(doseEvaluated,2)

        if doseEvaluated(i,j)>=Threshold

            currentGamma=sqrt(double((doseEvaluated(i,j)-doseReference(i,j))^2/dDose^2));
         
            k_radius=floor(currentGamma*dDistance+1);
            
			k=1;

            while k<=k_radius
                for kki=-k:k
                    for kkj=-k:k

                        if ((abs(kki)==k) || (abs(kkj)==k)) && (i+kki)>0 && (j+kkj)>0 && (i+kki)<size(doseReference,1) && (j+kkj)<size(doseReference,2) 
							newGamma=sqrt((doseEvaluated(i,j)-doseReference(i+kki,j+kkj))^2/dDose^2+(kki^2+kkj^2)/dDistance^2);
                            if newGamma<currentGamma
                                currentGamma=newGamma;
                            end
                        end %if
                        
                    end %kkj
                end %kki
                
                k=k+1;

            end %while
                 
            gammaMatrix(i,j)=currentGamma;

        end %if...Threshold
       
    end %j
end %i

N_fail=find(gammaMatrix>1);
N_unEval=find(gammaMatrix<0);
totalEvaluated=size(doseEvaluated,1)*size(doseEvaluated,2)-length(N_unEval);

pass_rate = (100-length(N_fail)/(totalEvaluated)*100);
% disp(['Fail rate =  ',num2str(N_fail),' of total evaluated ',num2str(totalEvaluated),' points: Pass rate [%] = ',num2str(100-N_fail/totalEvaluated*100)])

% 
% load gammaColormap
% figure
% imagesc(gammaMatrix)
% caxis([0 1])
% colormap(gammaColormap)
% colorbar

% timeElapsed=toc(timeStart);
% disp(['Gamma calculation done. Total calc time = ',num2str(timeElapsed),' s'])