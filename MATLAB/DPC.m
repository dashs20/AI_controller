% Define IC and cart properties
IC = [0,-1,deg2rad(20),0,deg2rad(-20),0]; % [x,dx,t1,dt1,t2,dt2]
cart_properties = [0.1  ,1,1,1,3]; % [m,m1,m2,l1,l2]
g = 9.81;
F = 0;

% define simulation options
op = odeset('abstol',1e-13,'reltol',1e-13); % propegation options
tmax = 10; % seconds
steps = tmax*100;
t_vec = linspace(0,tmax,steps);
[t,state] = ode45(@(t, state) DPC_dynamics_model(cart_properties,state,g,F), t_vec, IC, op);

% generate frames
obj = VideoWriter("cart_animation");
open(obj);
for i = 1:length(t)
    draw_cart(state(i,:),cart_properties,[-5*cart_properties(4),5*cart_properties(4)])
    writeVideo(obj,getframe)
    clf
end
close(obj)

function dstate = DPC_dynamics_model(cart_properties,state,g,F)
% CREDIT FOR MATH: TU Berlin

% unpact cart properties
m = cart_properties(1);
m1 = cart_properties(2);
m2 = cart_properties(3);
l1 = cart_properties(4);
l2 = cart_properties(5);

% unpact cart state
x = state(1);
dx = state(2);
t1 = state(3);
dt1 = state(4);
t2 = state(5);
dt2 = state(6);

% calculate M(y) matrix
My = [m + m1 + m1, l1*(m1 + m2)*cos(t1), m2*l2*cos(t2);...
      l1 * (m1 + m2) * cos(t1), l1^2 * (m1 + m2), l1 * l2 * m2 * cos(t1 - t2);...
      l2 * m2 * cos(t2), l1 * l2 * m2 * cos(t1 - t2), l2^2 * m2];

% calculate RHS
RHS1 = [l1 * (m1 + m2) * dt1^2 * sin(t1) + m2 * l2 * dt2^2 * sin(dt2);...
       -l1 * l2 * m2 * dt2^2 * sin(t1 - t2) + g * (m1 + m2) * l1 * sin(t1);...
        l1 * l2 * m2 * dt1^2 * sin(t1 - t2) + g * l2 * m2 * sin(t2)];
RHS2 = [F;0;0];

% calculate vector of derivatives
result = inv(My) * RHS1 + RHS2;
ddx = result(1);
ddt1 = result(2);
ddt2 = result(3);

dstate = [dx;ddx;dt1;ddt1;dt2;ddt2];

end

% animator function
function draw_cart(state,cart_properties,bounds)
L1 = cart_properties(4);
L2 = cart_properties(5);

% plot cart
p1 = [state(1) - L1/2,L1/2];
p2 = [state(1) + L1/2,L1/2];
p3 = [state(1) + L1/2,-L1/2];
p4 = [state(1) - L1/2,-L1/2];
cart_points = vertcat(p1,p2,p3,p4,p1);
plot(cart_points(:,1),cart_points(:,2));

hold on

% other stuff
yline(0)
xline(0)

% plot pendulum 1
p6 = [state(1) + L1 * sin(state(3)),L1 * cos(state(3))];
pend_points = vertcat([state(1),0],p6,[state(1),0]);
plot(pend_points(:,1),pend_points(:,2))

% plot pendulum 2
p7 = [p6(1) + L2 * sin(state(5)),p6(2) + L2 * cos(state(5))];
pend_points = vertcat(p6,p7,p6);
plot(pend_points(:,1),pend_points(:,2))

axis equal
xlim(bounds)
ylim(bounds)
end