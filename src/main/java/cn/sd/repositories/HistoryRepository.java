package cn.sd.services.repositories;

import cn.sd.entities.Detection;
import org.springframework.data.domain.Page;
import org.springframework.data.repository.PagingAndSortingRepository;

import org.springframework.data.domain.Pageable;

// JpaRepository实现简单查询
public interface HistoryRepository extends PagingAndSortingRepository<Detection, Long> {
//   @Query("from Detection d where d.userId=1")
   Page <Detection> findAllByUserId(Pageable pageable, Long userId);
    }
//    List<User> findByEmail(String email);

