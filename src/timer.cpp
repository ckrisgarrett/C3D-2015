/*
    File:   timer.cpp
    Author: Kris Garrett
    Date:   September 3, 2013
*/

#include "timer.h"


Timer::Timer()
{
    timeStarted = false;
    timeEnded = false;
    timeElapsed = 0.0;
}

void Timer::start()
{
    clock_gettime(CLOCK_MONOTONIC, &time1);
    timeStarted = true;
    timeEnded = false;
}


void Timer::restart()
{
    timeElapsed = 0.0;
    start();
}


void Timer::stop()
{
    clock_gettime(CLOCK_MONOTONIC, &time2);
    timeEnded = true;
    
    
    // Calculate elapsed time.
    if(timeStarted) {
        timespec diffTime;
        diffTime.tv_sec = time2.tv_sec - time1.tv_sec;
        diffTime.tv_nsec = time2.tv_nsec - time1.tv_nsec;
        
        if(diffTime.tv_nsec < 0) {
            diffTime.tv_sec--;
            diffTime.tv_nsec += 1E9;
        }
        
        timeElapsed += diffTime.tv_sec + diffTime.tv_nsec / 1E9;
    }
}

double Timer::getTimeElapsed()
{
    return timeElapsed;
}
