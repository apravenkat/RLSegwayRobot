#include "Motor.h"
#include "Balanced.h"


Timer2 Timer2;
extern Mpu6050 Mpu6050;
extern Motor Motor;
extern Balanced Balanced;



void setup() 
{
  Motor.Pin_init();
  //Motor.Encoder_init();
  Timer2.init(TIMER);
  
  Mpu6050.init();
  //Serial.begin(9600);
  //delay(100);
  Serial.begin(115200);
  delay(1000);

}

void loop() 
{
  int direction_buf[] = {FORWARD,BACK,LEFT,RIGHT,STOP};
  static unsigned long print_time;
  Balanced.Motion_Control(STOP);
  
}
