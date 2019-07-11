
document.addEventListener('DOMContentLoaded',()=>{

let dataB1 = [2,1,0];
let dataB2 = [3,1,0];
let dataB3 = [2,5,0];
let dataB4 = [1,1,0];

let dataR1 = [3,1.5,1];
let dataR2 = [3.5,0.5,1];
let dataR3 = [4,1.5,1];
let dataR4 = [5.5, 1,1];

//unknown
let dataU = [4.5,1,"it should be 1"];

let all_points = [dataB1,dataB2,dataB3,dataB4,dataR1,dataR2,dataR3,dataR4];

function sigmoid(x){
    return 1/(1+Math.exp(-x)) ;
}

//training
function train(){
    let w1 = Math.random()*0.2 -0.1;
    let w2 = Math.random()*0.2 -0.1;
    let b = Math.random()*0.2 -0.1;
    let learning_rate = 0.2;

    for (let iter=0; iter<50000; iter++){
        //pick random point
        let random_idx = Math.floor(Math.random()*all_points.length);
        let point = all_points[random_idx];
        let target = point[2];

        //feed forward
        let z = w1*point[0]+w2*point[1] + b;
        let pred = sigmoid(z);

        //compare model pred with actual value
        let cost = (pred - target)**2

        //find slope of cost w.r.t each parameter

        let dcost_pred = 2*(pred - target);

        let dpred_dz=sigmoid(z)*(1-sigmoid(z));

        let dz_dw1 = point[0];
        let dz_dw2 = point[1];
        let dz_db = 1;
         // chain rule
        let dcost_dw1 = dcost_pred*dpred_dz*dz_dw1;
        let dcost_dw2 = dcost_pred*dpred_dz*dz_dw2;
        let dcost_db = dcost_pred*dpred_dz*dz_db;
        //update params
        w1 -= learning_rate*dcost_dw1;
        w2 -= learning_rate*dcost_dw2;
        b -= learning_rate*dcost_db;
    }

    return {w1:w1, w2:w2, b:b};

    }

    
    
}


























})