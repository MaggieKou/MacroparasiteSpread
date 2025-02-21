ep_m =1e1;
ep_u =1e0;
ep_i=1e-3;
zeta=1e-4;
xi=1e-4;
gam_A=-1e1;
gam_m=1e1;
c=0.8;

%domain setup
x = linspace(0,1e1, 100); %nondimensionalized space
t = linspace(0,2.5, 100); % nondimensionalized time
sol_xt = zeros(length(t), length(x));
disp("CFL: " + c*(x(length(x)-x(1))*length(t)/(length(x)*(t(length(t))-t(1)))));

param = param_def(ep_m, ep_u, ep_i, zeta, xi, gam_A, gam_m, c);
U_init = 1e-5*ones(1,length(x));
I_init = 1e-5*ones(1,length(x)); %TODO: change back to zero

%initialize mean as a Gaussian with 
m_init = GaussianGen(x, 0, 1e-3, 0.1); 
mp_init = 1e-3*ones(1,length(x));
%take A to be a negative binomial - VMR = m/k + 1
A_init = m_init/0.8 + 1 ;
Ap_init = 1e-2*ones(1,length(x));

%x_init is two index: 1st variable, 2nd spatial
x_init = [U_init; I_init; m_init; -1/param(8)*mp_init; A_init; -1/param(8)*Ap_init]; 

sol_eta_x = solve_from_init(t, x, x_init, param);
sol_x_t = eta2time(x,t, sol_eta_x, x_init, param);

tiledlayout(2,1)

ax1 = nexttile;
hold on
plot(ax1,x,1e4*sol_x_t(1,:,3),'DisplayName', 'init');
plot(ax1,x,1e4*sol_x_t(round(length(t)/2),:,3),'DisplayName', 'halfway');
plot(ax1,x,1e4*sol_x_t(length(t),:,3),'DisplayName', 'final');
hold off
hl = legend('show');
set(hl, 'Interpreter','latex')
%set(gca, 'YScale', 'log')
set(gca,'TickLabelInterpreter','latex')
ylabel(ax1,"$\hat{m}$",'interpreter', 'latex') 
xlabel(ax1,"$x$",'interpreter', 'latex') 
title(ax1, 'mean for various times', 'Interpreter','latex')

ax2 = nexttile;
hold on
plot(ax2,x,sol_x_t(1,:,2),'DisplayName', 'init');
plot(ax2,x,sol_x_t(round(length(t)/2),:,2),'DisplayName', 'halfway');
plot(ax2,x,sol_x_t(length(t),:,2),'DisplayName', 'final');
hold off
hl = legend('show');
set(hl, 'Interpreter','latex')
set(gca,'TickLabelInterpreter','latex')
ylabel(ax2,"$\hat{I}$",'interpreter', 'latex') 
set(gca, 'YScale', 'log')
xlabel(ax2,"$x$",'interpreter', 'latex') 
title(ax2,'Abundance of free-living larvae', 'Interpreter','latex')

function dist = GaussianGen(x, x_o, A, var) %produces Gaussian Distribution
dist = A.*exp(-(x-x_o).^2./var);
end

function sol_eta_x = solve_from_init(t, x, init, param)
%takes intitial values for U, I, m, m', and A along x %init is 6 x N 
    sol_eta_x = zeros(length(t), length(init(1,:)), 4); %solution for rate of change of 4 variables in spacetime
disp("Now solving spatial points:")
for i = 1:length(init(1,:))
    disp(i + "/" + length(init(1,:)))
    eta_span = [x(i) - param(8)*t(1),x(i) - param(8)*t(length(t))];
    [eta,y_i] = ode23s(@(eta,y_i) myode(y_i, param), eta_span, init(:, i));
     
    y_new = zeros(length(t), 6);
    y_new(:,:) = interp1(eta, y_i(:,:), x(i) - param(8).*t); %interpolate to the dimensionality of t-space
    eta_new = x(i) - param(8).*t;
    d_eta = length(eta_new)/(eta_new(length(eta_new))-eta_new(1)); %step size in eta
    %TODO: explicitly track U and I primed
    U_eta = zeros(1,length(y_new(:, 1)));
    U_eta(2:end) = diff(y_new(:,1))./d_eta; %compute the derivative for U
    I_eta = zeros(1,length(y_new(:, 2))); 
    I_eta(2:end) = diff(y_new(:, 2))./d_eta; %compute the derivative for I

    %return values
    sol_eta_x(:, i, 1) = U_eta; %U_eta
    sol_eta_x(:, i, 2) = I_eta; %I_eta
    sol_eta_x(:, i, 3) = y_new(:, 4); %m_eta
    sol_eta_x(:, i, 4) = y_new(:, 6); %A_eta
    end

end
function sol_x_t = eta2time(x, t, sol_eta_x, init, param)
    sol_x_t = zeros(length(t), length(x), 4); %information about U, I, m, A
    dt = (t(length(t)) - t(1))/length(t);
    sol_x_t(1, :, 1) = init(1,:);
    sol_x_t(1, :, 2) = init(2,:);
    sol_x_t(1, :, 3) = init(3,:);
    sol_x_t(1, :, 4) = init(5,:);
    for j = 2:length(t)
        sol_x_t(j, :, :) = sol_x_t(j-1, :, :) - 1/2*param(8)*dt*(sol_eta_x(j-1,:,:)+sol_eta_x(j-1,:,:));
    end
end
function param = param_def(ep_m, ep_u, ep_i, zeta, xi, gam_A, gam_m, c)
   %ep_i and ep_u refer to the rates for the larval development stages,
   %gam_A and gam_m are the nondimensional control param

    param = [ep_m, ep_u, ep_i, zeta, xi, gam_A, gam_m, c];
end

function state_change = myode(sol, param) %function for full set of equations with uninfective stage

   %statechange is a 6 dimensional vector spanning the set of ODEs for the
   %receding frame d/dt([U, I, m, m', A, A']) which is a function of:
   %sol: [U, I, m, m', A, A']
   %param: [ep_u, ep_i, zeta, xi, gam_A, gam_m]
   %c:specified speed in the receding frame
    state_change = zeros(6,1);
    state_change(1) = -param(1)*sol(3)/param(8) + param(2)*sol(1)/param(8);
    %U' = -ep_U*m/c + ep_U*U/c;
    state_change(2) = -sol(1)/param(8) + param(3)*sol(2)/param(8);
    %I' = -U/c + ep_I*I/c
    state_change(3) = sol(4);
    %m'
    state_change(4) = -param(8)*sol(4) + sol(3)-sol(2)+sol(3).*sol(5) + sol(3).^2;
    %m'' = -cm' + m - I + m*A + m^2
    state_change(5) = sol(6);
    %A'
    state_change(6) = -param(8)*sol(6) - 2*sol(4).*(sol(6)+sol(4))./sol(3) - param(4) - sol(2).*(param(5)-sol(5))./sol(3) - param(6)*sol(5) - param(7)*sol(3) -sol(3).^2 + sol(5).^2;
    %A'' = -c*A' -2*(m'(A'+m'))/m - xi - I(chi - A)/m - gam_A*A - gam_m*m -
    %m^2 + A^2
    if state_change(3) > 100
        state_change(3) = 100;
        state_change(4) = 0;
    end
    
    if state_change(5) > 100
        state_change(5) = 100;
        state_change(6) = 0;
    end
    if state_change(5) < 0
        state_change(5) = 0;
        state_change(6) = 0;
    end
end


function stat_tot = state_integrator(sol_x_t, x, t)
%integrates over space for all time to find total values 
state_tot = zeros(length(t),4);
state_tot(1, :) = 0; 
dx = (x(length(x)) - x(1))/length(x); %spatial grid siz
%for loop through timesteps
for i = 1:length(t)
    for j = 2: length(x)
    state_tot(i, :) = dx*(sol_x_t(i, j, :) + sol_x_t(i, j-1, :))./2; %trapezoidal integration
    end
end
end