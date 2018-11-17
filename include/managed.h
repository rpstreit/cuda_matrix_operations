
#ifndef MANAGED_H
#define MANAGED_H

class Managed {
public:
  void *operator new(size_t len); 
  void operator delete(void *ptr);
}

#endif
