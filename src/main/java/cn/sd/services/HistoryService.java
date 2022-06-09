package cn.sd.services;

import cn.sd.entities.NewDetection;
import cn.sd.repositories.HistoryRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;

import javax.persistence.EntityManager;
import javax.persistence.PersistenceContext;

@Service
public class HistoryService {

    private HistoryRepository historyRepository;
    @PersistenceContext
    private EntityManager em;

    @Autowired
    public HistoryService(HistoryRepository historyRepository) {
        this.historyRepository = historyRepository;
    }


    public Page getObjectivePageable(Pageable pageable, Long userId) {

        Page<NewDetection> objectivePage = getObjectivePage(pageable, userId);
        return objectivePage;
    }

    public Page<NewDetection> getObjectivePage(Pageable pageable, Long userId) {
//        CriteriaBuilder criteriaBuilder =em.getCriteriaBuilder();
//        CriteriaQuery<Detection> criteriaQuery = criteriaBuilder.createQuery(Detection.class);
//        Root<Detection> detection = criteriaQuery.from(Detection.class);

        return historyRepository.findAllByUserId(pageable, userId);

//                (Specification<Detection>)(root, criteriaQuery , criteriaBuilder) -> {

//
//           Predicate predicate = criteriaBuilder.equal(root.get("userId"),userId);
//           // List <Predicate> predicates=new ArrayList<Predicate>();
//          //  predicates.add(predicate);
//                    //=HistoryService.getneedPredicates(userId, root ,criteriaBuilder);
//         //   criteriaQuery.where(predicates);
//            return  criteriaQuery.where(predicate);
//           // .toArray(new Predicate[predicates.size()])
//            });
    }
}
