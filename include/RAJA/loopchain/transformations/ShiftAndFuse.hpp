// Contains code for the shift and fuse transformation, including the analysis required for it

#ifndef RAJA_LCSHIFTFUSE_HPP
#define RAJA_LCSHIFTFUSE_HPP

#include "RAJA/config.hpp"
#include "RAJA/loopchain/KernelWrapper.hpp"
#include "RAJA/loopchain/Chain.hpp"
#include "RAJA/loopchain/transformations/Common.hpp"
#include "RAJA/loopchain/transformations/Shift.hpp"
#include "RAJA/loopchain/transformations/Fuse.hpp"

namespace RAJA {

//Transformation Declarations

template <typename... KernelTypes>
auto shift_and_fuse(camp::tuple<KernelTypes...> knlTuple);

template <typename... KernelTypes>
auto shift_and_fuse(KernelTypes... knls);


//Analysis Declarations

template <typename... KernelTypes>
auto can_fuse(camp::tuple<KernelTypes...> knlTuple);

template <typename... KernelTypes>
auto can_fuse(KernelTypes... knls);




//Analysis Definitions

//returns the constraints on the shift values created from the dependences between the two kernels. The kernel id values are used in the shift amounts
std::string shift_constraints_for_pair(auto knl1, auto knl2, auto id1, auto id2 ) {

  constexpr int numDims = knl1.numArgs;
  
  assert(knl1.numArgs == knl2.numArgs);

  isl_ctx * ctx = isl_ctx_alloc();

  isl_union_map * depRelation = data_dep_relation(ctx, knl1, knl2, id1, id2);

  isl_union_set * ispace1 = knl_iterspace(ctx, knl1, id1);

  //If there are no dependences from knl1 to knl2, there are no constraints between these two.
  auto maxInput = isl_union_set_sample_point(isl_union_set_lexmax(ispace1));
  auto image = isl_union_set_apply(isl_union_set_from_point(isl_point_copy(maxInput)), depRelation);
  if(isl_union_set_is_empty(image)) {
    return "true";
  }

  //now we want to get the minimum value in each dimension of the image elements;
  
  int * minValues = new int[numDims+1];
  minValues[0] = numDims;
  for(int i = 1; i <= numDims; i++) { minValues[i] = std::numeric_limits<int>::max(); }

  int * inputPointValues = new int[numDims+1];
  inputPointValues[0] = numDims;
  for(int i = 1; i <= numDims; i++) {inputPointValues[i] = std::numeric_limits<int>::max(); }

  auto update_min_values = [] (isl_point * p, void * currMin_) {
    int * currMin = (int *) (currMin_);
    int numDims = currMin[0];
    for(int i = 0; i < numDims; i++) {
      isl_val * pointVal = isl_point_get_coordinate_val(p, isl_dim_set, i);
      int value = isl_val_get_num_si(pointVal);
      //std::cout << "point value: " << value << "\n";
      if(currMin[i+1] > value) {
        currMin[i+1] = value;
        //std::cout << "changed\n";
      }
    }
    return isl_stat_ok;
  }; //update_min_values
 //collect the point values for the input point
  update_min_values(maxInput, (void*) inputPointValues);
  //collect the minimum values for the output points
  isl_union_set_foreach_point(image, update_min_values, (void*) minValues);
  for(int i = 0; i < numDims; i++) {
   // std::cout << "dim " << i << " input: " << inputPointValues[i+1] << "\n";
   // std::cout << "output minimum " << i << " is " << minValues[i+1]<< "\n";;
  }

  //Now that we have the output minumums for the input point, we need to set up the constraint.
  //For each dimension, we have input - outputMax = shift2 - shift1
  std::string constraints = "";
  for(int i = 0; i < numDims; i++) {
    std::string constraint = "";
    constraint += std::to_string(inputPointValues[i+1]) + "-" + std::to_string(minValues[i+1]) + "=";
    constraint += "S" + std::to_string(id2) + "_" + std::to_string(i);
    constraint += " - ";
    constraint += "S" + std::to_string(id1) + "_" + std::to_string(i);
    //std::cout << "Constraint " << i << ": " << constraint << "\n";;
   constraints += constraint;
    if (i != numDims - 1) {constraints += " and ";}
  }
  //std::cout << "Constraints: " << constraints << "\n";

  return constraints;
} // shift_constraints_for_pair

// Recurses across the pairs of kernels to generate and concatenate 
// the constraints between each pair. 
template <camp::idx_t KnlId1, camp::idx_t KnlId2, camp::idx_t NumKnls>
auto shift_constraints_helper(isl_ctx * ctx, auto knlTuple) {

  if constexpr (KnlId1 == NumKnls) {
    //base case
    std::string terminal = "END";
    return terminal;
  } else if constexpr (KnlId2 == NumKnls) {
    //iterate to the next source kernel, 
    //make the destination kernel the one following it
    return shift_constraints_helper<KnlId1+1, KnlId1+2, NumKnls>(ctx, knlTuple);
  } else {
    // calculate the constraints between the shifts for the two kernels
    auto constraintString = shift_constraints_for_pair(camp::get<KnlId1>(knlTuple), 
                                                       camp::get<KnlId2>(knlTuple), 
                                                       KnlId1, KnlId2);

    auto rest = shift_constraints_helper<KnlId1, KnlId2+1, NumKnls>(ctx, knlTuple);

    if(rest == "END") {
      return constraintString;
    } else if (constraintString == "") {
      return rest;
    } else {
      return constraintString + " and " + rest;
    }
  }

}


// Returns the string containing the constraints on shift amounts using the dependences between the kernels
template <typename...KernelTypes, camp::idx_t...Is>
auto shift_constraints(isl_ctx * ctx, camp::tuple<KernelTypes...> knlTuple, camp::idx_seq<Is...> seq) {
  return shift_constraints_helper<0,1,sizeof...(Is)>(ctx, knlTuple);
} //shift_constraint


//Returns an isl_set containing all valid shift amounts combinations for knlTuple 
template <typename... KernelTypes, camp::idx_t... Is>
auto valid_shift_set(camp::tuple<KernelTypes...> knlTuple, camp::idx_seq<Is...> seq) {
  constexpr int numKernels = sizeof...(Is);
  constexpr int numDims = camp::get<0>(knlTuple).numArgs;

  isl_ctx * ctx = isl_ctx_alloc();
  isl_printer * p = isl_printer_to_file(ctx, stdout);
  auto deprel = dependence_relation_from_kernels(ctx, knlTuple, seq);


  //std::cout << "Dependence relation among kernels\n";
  //p= isl_printer_print_union_map(p,deprel);
  //std::cout << "\n";

  //First, we create the string for the shift variables and the constraint that they are all greater than = 0.
  std::string shiftSet = "[";
  std::string nonNegConstraint = "";
  for(int loopNum = 0; loopNum < numKernels; loopNum++) {
    for(int dim = 0; dim < numDims; dim++) {
      std::string shiftString = "S" + std::to_string(loopNum) + "_" + std::to_string(dim);

      shiftSet += shiftString;
      nonNegConstraint += shiftString + " >= 0 ";

      if(dim != numDims - 1) {
       shiftSet += ",";
       nonNegConstraint += " and ";
      }
    }//dim
    if(loopNum != numKernels - 1) {
      shiftSet += ",";
      nonNegConstraint += " and ";
    }
  }//loopNum

  shiftSet += "]";
 
  auto constraints = shift_constraints( ctx, knlTuple, seq);

  std::string constrainedShiftSetString = "{" + shiftSet + " : " + nonNegConstraint + " and " + constraints + "}";

  isl_set * possibleShifts = isl_set_read_from_str(ctx, constrainedShiftSetString.c_str());

  return possibleShifts;
}//valid_shift_set


//returns false if there is no valid shifting that enables fusion
template <typename... KernelTypes, camp::idx_t...Is>
auto can_fuse(camp::tuple<KernelTypes...> knlTuple, camp::idx_seq<Is...> seq) {
  isl_set * validShifts = valid_shift_set(knlTuple, seq);
  return ! isl_set_is_empty(validShifts);
}

template <typename... KernelTypes>
auto can_fuse(camp::tuple<KernelTypes...> knlTuple) {
  
  return can_fuse(knlTuple, idx_seq_for(knlTuple));
}

template <typename... KernelTypes>
auto can_fuse(KernelTypes... knls) {
  auto knlTuple = make_tuple(knls...);
  return can_fuse(knlTuple, index_seq_for(knlTuple));
}


//Transformation Definitions

// Converts a linear vector of shift amounts into the correct shape.
// Results in a tuple of tuples, where each inner tuple contains 
// the shift amounts for an individual kernel
template <camp::idx_t NumKnls, camp::idx_t...Dims>
auto shift_vector_to_shift_tuples(std::vector<int> shiftVector, camp::idx_seq<Dims...> dimSeq) {
  if constexpr (NumKnls == 0) {
    return make_tuple();
  } else {
    auto currTuple = make_tuple(shiftVector.at(Dims)...);
    std::vector<int> remainingShifts = {};
    for(int i = sizeof...(Dims); i < shiftVector.size(); i++) {
      remainingShifts.push_back(shiftVector.at(i));
    }
    auto restTuple = shift_vector_to_shift_tuples<NumKnls-1>(remainingShifts, dimSeq);

    return tuple_cat(make_tuple(currTuple), restTuple);
  }
}

template <camp::idx_t NumKnls, camp::idx_t NumDims>
auto shift_vector_to_shift_tuples(std::vector<int> shiftVector) {
  return shift_vector_to_shift_tuples<NumKnls>(shiftVector, camp::make_idx_seq_t<NumDims>{});
}

// Assuming there is a valid shift, calculates it and formats it into tuples
template <typename... KernelTypes, camp::idx_t...Is>
auto shift_amount_tuples(camp::tuple<KernelTypes...> knlTuple, camp::idx_seq<Is...> seq) {
  
  constexpr int numKernels = sizeof...(KernelTypes);
  constexpr int numDims = camp::get<0>(knlTuple).numArgs;

  auto validShifts = valid_shift_set(knlTuple, seq);

  isl_point * smallestShifts = isl_set_sample_point(isl_set_lexmin(validShifts));

  // The smallest shift is a linear sequence of all the numKnls * numDims shift
  // amounts, so we extract it into a vector, then convert that to the tuple of tuples

  auto point_to_vector = [](isl_point * p, int numDims) {
    std::vector<int> vals = {};

    for(int i = 0; i < numDims; i++) {
      isl_val * dimVal = isl_point_get_coordinate_val(p, isl_dim_set, i);
      vals.push_back(isl_val_get_num_si(dimVal));
    }
    return vals;
  };

  auto shiftAmounts = point_to_vector(smallestShifts, numKernels * numDims);

  return shift_vector_to_shift_tuples<numKernels,numDims>(shiftAmounts);
}//shift_amount_tuples


template <camp::idx_t...Is>
auto zip_shift(auto knlTuple, auto shiftAmountTuple, camp::idx_seq<Is...>) {
  return make_tuple(shift(camp::get<Is>(knlTuple), camp::get<Is>(shiftAmountTuple))...);
}

template <typename... KernelTypes, camp::idx_t...Is>
auto shift_and_fuse(camp::tuple<KernelTypes...> knlTuple, camp::idx_seq<Is...> seq) {
  if( can_fuse(knlTuple) ) {
    
  } else {
    std::cerr << "Need to create the padding empty kernels\n";
  }
  auto shiftAmountTuples = shift_amount_tuples(knlTuple, seq);
    auto shiftedKnls = make_tuple((shift(camp::get<Is>(knlTuple), camp::get<Is>(shiftAmountTuples)))...);
    auto fused = fuse(shiftedKnls);
    return fused;
}
template <typename... KernelTypes>
auto shift_and_fuse(camp::tuple<KernelTypes...> knlTuple) {
  return shift_and_fuse(knlTuple, idx_seq_for(knlTuple));
}

template <typename... KernelTypes>
auto shift_and_fuse(KernelTypes... knls) {
  auto knlTuple = make_tuple(knls...);
  return shift_and_fuse(knlTuple, idx_seq_for(knlTuple));
}




 }//namespace RAJA


#endif
