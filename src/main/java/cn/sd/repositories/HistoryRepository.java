package cn.sd.repositories;

import cn.sd.entities.NewDetection;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.repository.PagingAndSortingRepository;

// JpaRepository实现简单查询
public interface HistoryRepository extends PagingAndSortingRepository<NewDetection, Long> {
    //   @Query("from Detection d where d.userId=1")
    Page<NewDetection> findAllByUserId(Pageable pageable, Long userId);
}
//    List<User> findByEmail(String email);

