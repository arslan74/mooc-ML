% matrix with 3 rows and 3 colums 
A = [ 1, 1, 2; 3, 5, 8; 13, 21, 34 ]

% multiplication by scalar value
C = A * 3;
% addition and subtraction
  % create matrix rows using spaces
B = [1 1 1;2 2 2; 3 3 3];
  % addition
C = A + B;
  % substraction
C = A - B;

% transpose of a matrix
AT = A';

% common vectors , creates matrix of ones 
%U = ones(3,1);
U = ones(3,5);

diag(A);
diag(diag(A),0);

% identity matrix
eye(4);

% systematric matric
systematric = [2,1,5;1,3,4; 5 4 -2];
systematric';

% inverse matrix
inv(systematric);
%determinant
det(systematric);

% rows and columns
rows(A);
columns(U);

% sum of matrix
sum(U);
sum(sum(U));
mean(A);

% for loops 

