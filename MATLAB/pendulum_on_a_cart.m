% options
Fopt = 0;
m = 1;
M = 2;
g = 9.81;
L = 0.25;
IC = [-L,0,deg2rad(5),0,0,0]; % 5th thing is position integral IV

% simulate
tvec = linspace(0,10,1000);
op = odeset('abstol',1e-13,'reltol',1e-13);
[t,state] = ode45(@(t, state) dstate_calc(state,m,M,g,L),tvec,IC,op);

% % plot
% subplot(2,1,1)
% plot(t,state(:,1))
% 
% subplot(2,1,2)
% plot(t,state(:,3))

% plot(t,state(:,5))

% generate frames
obj = VideoWriter("cart_animation");
open(obj);
for i = 1:length(t)
    draw_cart(state(i,:),L,[-L*2,L*2])
    writeVideo(obj,getframe)
    clf
end
close(obj)

% dynamics model
function dstate = dstate_calc(state,m,M,g,L)

a = state(1); % x
b = state(2); % x dot
c = state(3); % theta
d = state(4); % theta dot
e = state(5); % integral of position of cart
f = state(6); % integral of angle of pendulum

% pd controller
Pa = 0;
Ia = 0;
Da = 0;

Pp = -500;
Ip = 0;
Dp = -5;

Fopt_angle = Pa * c + Ia * f + Da * d;
Fopt_position = Pp * a + Ip * e + Dp * b;

Fopt = Fopt_angle + Fopt_position;

da = b;
dc = d;
db = (2 * Fopt - m * g * cos(c) * sin(c) - 2 * d^2 * m * L * sin(c))/...
     (2 * (M + m) + m * cos(c)^2);
dd = (g * sin(c) - db * cos(c))/...
     (2 * L);
ie = e + a;
int_f = f + c;

dstate = [da;db;dc;dd;ie;int_f];
end

% animator function
function draw_cart(state,L,bounds)
% plot cart
p1 = [state(1) - L/2,L/2];
p2 = [state(1) + L/2,L/2];
p3 = [state(1) + L/2,-L/2];
p4 = [state(1) - L/2,-L/2];
cart_points = vertcat(p1,p2,p3,p4,p1);
plot(cart_points(:,1),cart_points(:,2));

hold on

% other stuff
yline(0)
xline(0)

% plot pendulum
p6 = [state(1) + L * sin(state(3)),L * cos(state(3))];
pend_points = vertcat([state(1),0],p6,[state(1),0]);
plot(pend_points(:,1),pend_points(:,2))

% plot point mass
plot(p6(1),p6(2),"r.")
axis equal
xlim(bounds)
ylim(bounds)
end