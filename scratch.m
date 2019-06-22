
trivsol = 1;
gbest = 2;
gambest = gams(gbest, :)

while trivsol == 1
    [DVs, ~, ~, ~, ~] = SZVD(Atrain, gambest, D, 0, 0, Pt, maxits, beta, 0);
    
     % Round small entries to zero.
     DVs = DVs.*(ceil(abs(DVs) - ztol));
     
     % Check for trivial solution.
     if nnz(DVs) == 0
         % If trivial solution, update gbest by one and update gambest.
         gbest = gbest + 1;
         gambest = gams(gbest);
     else
         % Have found a nontrivial solution.
         trivsol = 0;
     end
end 

%% B

B4 = DVs;