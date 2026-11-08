#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace strata::core {
namespace fake_cache {inline bool fail_open=false,fail_admit=false,fail_fill=false,fail_verify=false;
 inline int opens=0,closes=0,admitted=0,fills=0,verifies=0;inline bool live=false;}
class ExpertCache{
 int64_t slots_=0,used_=0,blob_=0;
 std::vector<uint8_t> bytes_;
public:
 bool open(int64_t slots,int64_t layers,int64_t experts,int64_t blob,std::string& err){++fake_cache::opens;
  if(fake_cache::fail_open||slots<=0||layers<=0||experts<=0||blob<=0){err="cache open failed";return false;}
  slots_=slots;blob_=blob;bytes_.resize(static_cast<size_t>(slots*blob));fake_cache::live=true;return true;}
 int64_t slots() const{return slots_;}
 int32_t admit(int64_t,int64_t){if(fake_cache::fail_admit)return -1;++fake_cache::admitted;return used_<slots_?static_cast<int32_t>(used_++):-1;}
 bool fill_slot_blocking(int32_t n,const uint8_t* src,std::string& err,int64_t bytes){++fake_cache::fills;
  if(fake_cache::fail_fill||!src||n<0||n>=slots_||bytes!=blob_){err="fill failure";return false;}
  for(int64_t j=0;j<blob_;++j)bytes_[static_cast<size_t>(n*blob_+j)]=src[j];
  return true;}
 bool verify_slot(int32_t n,const uint8_t* src,std::string& err,int64_t bytes){++fake_cache::verifies;
  if(fake_cache::fail_verify||!src||n<0||n>=slots_||bytes!=blob_){err="verify failure";return false;}
  for(int64_t j=0;j<blob_;++j)if(bytes_[static_cast<size_t>(n*blob_+j)]!=src[j]){err="contents differ";return false;}
  return true;}
 const uint8_t* device_slot(int32_t n) const {return n>=0&&n<slots_?bytes_.data()+static_cast<size_t>(n*blob_):nullptr;}
 void close(){if(fake_cache::live){++fake_cache::closes;fake_cache::live=false;}bytes_.clear();slots_=0;used_=0;}
 ~ExpertCache(){close();}
};
}
