/*
    File:   timer.h
    Author: Kris Garrett
    Date:   September 3, 2013
*/

#ifndef __TIMER_H
#define __TIMER_H

#include <time.h>

class Timer
{
public:
    Timer();
    void start();
    void restart();
    void stop();
    double getTimeElapsed();
    
private:
    timespec time1, time2;
    bool timeStarted, timeEnded;
    double timeElapsed;
};

#endif
