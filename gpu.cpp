#include <iostream>
#include <SDL2/SDL.h>
#include <math.h>
#include <ctime>

#include <fstream>
#include <sstream>

#include </usr/include/CL/opencl.hpp>

constexpr int BOID_AMOUNT = 5000;
constexpr int WIDTH = 1280;
constexpr int HEIGHT = 960;

#define UPDATE_BOID(i, x,y,dx,dy) do {boidX[i] = x; boidY[i] = y;} while(0)

void drawSquare(SDL_Renderer* const ren, const int x, const int y)
{
        SDL_Rect rect;
        rect.x = x;
        rect.y = y;
        rect.w = 4;
        rect.h = 4;

        SDL_RenderFillRect(ren, &rect);
}

int main(int argc, char** argv)
{
    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window *win = SDL_CreateWindow("Boids", 100, 100, WIDTH, HEIGHT, SDL_WINDOW_SHOWN);
    SDL_Renderer *ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    
    double boidX[BOID_AMOUNT];
    double boidY[BOID_AMOUNT];
    double boidDx[BOID_AMOUNT];
    double boidDy[BOID_AMOUNT];

    srand(time(NULL));

    for (int i = 0; i < BOID_AMOUNT; i++)
    {
        int x = rand() % (WIDTH - 100) + 50;
        int y = rand() % (HEIGHT - 100) + 50;
        int dx = rand() % 10 - 5;
        int dy = rand() % 10 - 5;

        UPDATE_BOID(i, x, y, dx, dy);
    }

    /* -- */

    //get all platforms (drivers)
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(all_platforms.size()==0){
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform=all_platforms[2];
    std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";
 
    //get default device of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Device default_device=all_devices[0];
    std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";
 
 
    cl::Context context({default_device});
 
    cl::Program::Sources sources;
 
    // kernel calculates for each element C=A+B

    std::ifstream file("./kernel.cl");
    std::stringstream buffer;
    buffer << file.rdbuf();

    std::string kernel_code = buffer.str();

    sources.push_back({kernel_code.c_str(),kernel_code.length()});
 
    cl::Program program(context,sources);
    if(program.build({default_device})!=CL_SUCCESS){
        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)<<"\n";
        exit(1);
    }

    cl::Buffer bBoidX(context, CL_MEM_READ_WRITE, sizeof(double)*BOID_AMOUNT);
    cl::Buffer bBoidY(context, CL_MEM_READ_WRITE, sizeof(double)*BOID_AMOUNT);
    cl::Buffer bBoidDx(context, CL_MEM_READ_WRITE, sizeof(double)*BOID_AMOUNT);
    cl::Buffer bBoidDy(context, CL_MEM_READ_WRITE, sizeof(double)*BOID_AMOUNT);
    
    cl::CommandQueue queue(context,default_device);

    queue.enqueueWriteBuffer(bBoidX,CL_FALSE,0,sizeof(double)*BOID_AMOUNT,boidX);
    queue.enqueueWriteBuffer(bBoidY,CL_FALSE,0,sizeof(double)*BOID_AMOUNT,boidY);
    queue.enqueueWriteBuffer(bBoidDx,CL_FALSE,0,sizeof(double)*BOID_AMOUNT,boidDx);
    queue.enqueueWriteBuffer(bBoidDy,CL_TRUE,0,sizeof(double)*BOID_AMOUNT,boidDy);

    queue.flush();

    cl::Kernel kernel_move=cl::Kernel(program,"move");
    kernel_move.setArg(0,bBoidX);
    kernel_move.setArg(1,bBoidY);
    kernel_move.setArg(2,bBoidDx);
    kernel_move.setArg(3,bBoidDy);

    /* -- */

    SDL_Event event;
    bool quit = false;
    while (!quit)
    {
        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_QUIT)
            {
                quit = true;
            }
        }

        int startTime = SDL_GetTicks();

        SDL_RenderClear(ren);

        SDL_SetRenderDrawColor(ren, 255, 255, 255, 255);

        queue.enqueueReadBuffer(bBoidX,CL_FALSE,0,sizeof(double)*BOID_AMOUNT,boidX);
        queue.enqueueReadBuffer(bBoidY,CL_FALSE,0,sizeof(double)*BOID_AMOUNT,boidY);

        queue.enqueueNDRangeKernel(kernel_move,cl::NullRange,cl::NDRange(BOID_AMOUNT),cl::NullRange);

        queue.flush();

        for (int i = 0; i < BOID_AMOUNT; i++)
        {
            drawSquare(ren, boidX[i], boidY[i]);
        }

        SDL_SetRenderDrawColor(ren, 24, 24, 24, 255);

        SDL_RenderPresent(ren);

        int frameTime = SDL_GetTicks() - startTime;
        double fps = (frameTime > 0) ? (1000.0 / frameTime) : 0;

        if (fps < 160)
        {
            std::cout << "FPS:" << fps << "\n";
        }
    }
    
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);

    SDL_Quit();
}
