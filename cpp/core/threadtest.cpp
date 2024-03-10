#include "../core/threadtest.h"

#include "../core/global.h"
#include "../core/multithread.h"
#include "../core/threadsafecounter.h"
#include "../core/threadsafequeue.h"
#include "../core/rand.h"
#include "../core/test.h"

//------------------------
#include "../core/using.h"
//------------------------

void ThreadTest::runTests() {
  cout << "Running thread tests" << endl;
  {
    WaitableFlag flag;
    ThreadSafeCounter counter0;
    ThreadSafeCounter counter1;
    ThreadSafeCounter counter2;
    ThreadSafeQueue<int> queue;
    ThreadSafeQueue<int> queue2;
    ThreadSafeCounter exitCount;

    counter0.add(4);
    counter1.add(2);
    counter2.add(8);
    auto f = [&]() {
      flag.waitUntilFalse();
      std::this_thread::yield();
      testAssert(queue.forcePush(10));
      counter0.add(-4);
      counter1.add(-4);
      counter2.add(-4);
      flag.waitUntilTrue();
      std::this_thread::yield();
      testAssert(queue.forcePush(11));
      counter0.add(2);
      counter1.add(2);
      counter2.add(2);
      flag.waitUntilFalse();
      std::this_thread::yield();
      testAssert(queue.forcePush(12));
      counter0.add(-6);
      counter1.add(-6);
      counter2.add(-6);
      flag.waitUntilTrue();
      testAssert(queue.forcePush(13));
      std::this_thread::yield();
      testAssert(queue.forcePush(14));
      std::this_thread::yield();
      testAssert(queue.forcePush(15));
      flag.waitUntilFalse();
      std::this_thread::yield();
      queue.setReadOnly();
      exitCount.add(1);
    };
    auto c0 = [&]() {
      counter0.waitUntilZero();
      testAssert(queue.forcePush(16));
      exitCount.add(1);
    };
    auto c1 = [&]() {
      counter1.waitUntilZero();
      testAssert(queue.forcePush(17));
      exitCount.add(1);
    };
    auto c2 = [&]() {
      counter2.waitUntilZero();
      testAssert(queue.forcePush(18));
      exitCount.add(1);
    };
    auto g = [&]() {
      int buf;
      testAssert(queue.waitPop(buf));
      testAssert(buf == 10);
      testAssert(queue.waitPop(buf));
      testAssert(buf == 16);
      flag.set(true);
      testAssert(queue.waitPop(buf));
      testAssert(buf == 11);
      testAssert(queue.waitPop(buf));
      testAssert(buf == 17);
      testAssert(!queue.tryPop(buf));
      std::this_thread::yield();
      testAssert(!queue.tryPop(buf));
      flag.set(false);
      testAssert(queue.waitPop(buf));
      testAssert(buf == 12);
      testAssert(queue.waitPop(buf));
      testAssert(buf == 18);
      flag.set(true);
      testAssert(queue.waitPop(buf));
      testAssert(buf == 13);
      while(!queue.tryPop(buf))
        std::this_thread::yield();
      testAssert(buf == 14);
      while(!queue.tryPop(buf))
        std::this_thread::yield();
      testAssert(buf == 15);
      queue2.close();
      testAssert(queue.waitPop(buf));
      testAssert(buf == 19);
      flag.setPermanently(false);
      flag.set(true);
      testAssert(!queue.waitPop(buf));
      exitCount.add(1);
    };
    auto h = [&]() {
      int buf;
      testAssert(!queue2.waitPop(buf));
      testAssert(queue.forcePush(19));
      exitCount.add(1);
    };

    std::vector<std::thread> threads;
    threads.push_back(std::thread(f));
    threads.push_back(std::thread(c0));
    threads.push_back(std::thread(c1));
    threads.push_back(std::thread(c2));
    threads.push_back(std::thread(g));
    threads.push_back(std::thread(h));

    for(std::thread& thread: threads) {
      thread.join();
    }
    exitCount.add(-6);
    exitCount.waitUntilZero();
    flag.set(true);
    flag.waitUntilFalse();
    testAssert(!queue.waitPush(20));
    testAssert(!queue2.waitPush(21));
    testAssert(!queue.isClosed());
    testAssert(queue2.isClosed());
    queue.close();
    testAssert(queue.isClosed());
  }

  {
    auto writer = [&](ThreadSafeQueue<int>* queue, double yieldProb) {
      Rand rand;
      for(int i = 1; i <= 10000; i++) {
        if(rand.nextBool(yieldProb))
          std::this_thread::yield();
        queue->waitPush(i);
      }
    };
    std::atomic<int64_t> total(0);
    std::atomic<int64_t> totalSq(0);
    auto reader = [&](ThreadSafeQueue<int>* queue, double yieldProb) {
      Rand rand;
      int64_t sum = 0;
      int64_t sumSq = 0;
      std::vector<int> buf;
      while(true) {
        if(rand.nextBool(yieldProb))
          std::this_thread::yield();
        if(rand.nextBool(0.5)) {
          int x;
          bool suc = queue->waitPop(x);
          if(!suc)
            break;
          sum += x;
          sumSq += x * x;
        }
        else {
          int n = rand.nextInt(1,8);
          bool suc = queue->waitPopUpToN(buf,n);
          if(!suc)
            break;
          assert(buf.size() > 0 && buf.size() <= n);
          for(int x: buf) {
            sum += x;
            sumSq += x * x;
          }
          buf.clear();
        }
      }
      total.fetch_add(sum);
      totalSq.fetch_add(sumSq);
    };

    {
      total.store(0);
      totalSq.store(0);
      ThreadSafeQueue<int> queue;
      std::vector<std::thread> writers;
      std::vector<std::thread> readers;
      writers.push_back(std::thread(writer,&queue,0.50));
      writers.push_back(std::thread(writer,&queue,0.40));
      writers.push_back(std::thread(writer,&queue,0.30));
      writers.push_back(std::thread(writer,&queue,0.25));
      writers.push_back(std::thread(writer,&queue,0.20));
      writers.push_back(std::thread(writer,&queue,0.15));
      writers.push_back(std::thread(writer,&queue,0.12));
      writers.push_back(std::thread(writer,&queue,0.10));
      readers.push_back(std::thread(reader,&queue,0.30));
      for(std::thread& thread: writers)
        thread.join();
      queue.setReadOnly();
      for(std::thread& thread: readers)
        thread.join();
      testAssert(total.load() == 8LL * 10000LL * 10001LL / 2);
      testAssert(totalSq.load() == 8LL * 10000LL * 10001LL * 20001LL / 6);
    }
    {
      total.store(0);
      totalSq.store(0);
      ThreadSafeQueue<int> queue;
      std::vector<std::thread> writers;
      std::vector<std::thread> readers;
      writers.push_back(std::thread(writer,&queue,0.50));
      writers.push_back(std::thread(writer,&queue,0.10));
      readers.push_back(std::thread(reader,&queue,0.50));
      readers.push_back(std::thread(reader,&queue,0.45));
      readers.push_back(std::thread(reader,&queue,0.40));
      readers.push_back(std::thread(reader,&queue,0.35));
      readers.push_back(std::thread(reader,&queue,0.30));
      readers.push_back(std::thread(reader,&queue,0.25));
      readers.push_back(std::thread(reader,&queue,0.20));
      readers.push_back(std::thread(reader,&queue,0.15));
      for(std::thread& thread: writers)
        thread.join();
      queue.setReadOnly();
      for(std::thread& thread: readers)
        thread.join();
      testAssert(total.load() == 2LL * 10000LL * 10001LL / 2);
      testAssert(totalSq.load() == 2LL * 10000LL * 10001LL * 20001LL / 6);
    }
  }
}
