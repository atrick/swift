//===--- LifetimeDependence.cpp -----------------------------------------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "swift/AST/LifetimeDependence.h"
#include "swift/AST/ASTContext.h"
#include "swift/AST/ConformanceLookup.h"
#include "swift/AST/Decl.h"
#include "swift/AST/DiagnosticsSema.h"
#include "swift/AST/Module.h"
#include "swift/AST/ParameterList.h"
#include "swift/AST/SourceFile.h"
#include "swift/AST/Type.h"
#include "swift/AST/TypeRepr.h"
#include "swift/Basic/Assertions.h"
#include "swift/Basic/Defer.h"
#include "swift/Basic/Range.h"

namespace swift {

LifetimeEntry *
LifetimeEntry::create(const ASTContext &ctx, SourceLoc startLoc,
                      SourceLoc endLoc, ArrayRef<LifetimeDescriptor> sources,
                      std::optional<LifetimeDescriptor> targetDescriptor) {
  unsigned size = totalSizeToAlloc<LifetimeDescriptor>(sources.size());
  void *mem = ctx.Allocate(size, alignof(LifetimeEntry));
  return new (mem) LifetimeEntry(startLoc, endLoc, sources, targetDescriptor);
}

std::optional<LifetimeDependenceInfo>
getLifetimeDependenceFor(ArrayRef<LifetimeDependenceInfo> lifetimeDependencies,
                         unsigned index) {
  for (auto dep : lifetimeDependencies) {
    if (dep.getTargetIndex() == index) {
      return dep;
    }
  }
  return std::nullopt;
}

std::string LifetimeDependenceInfo::getString() const {
  std::string lifetimeDependenceString = "@lifetime(";
  auto addressable = getAddressableIndices();
  
  auto getSourceString = [&](IndexSubset *bitvector, StringRef kind) {
    std::string result;
    bool isFirstSetBit = true;
    for (unsigned i = 0; i < bitvector->getCapacity(); i++) {
      if (bitvector->contains(i)) {
        if (!isFirstSetBit) {
          result += ", ";
        }
        result += kind;
        if (addressable && addressable->contains(i)) {
          result += "address ";
        }
        result += std::to_string(i);
        isFirstSetBit = false;
      }
    }
    return result;
  };
  if (inheritLifetimeParamIndices) {
    assert(!inheritLifetimeParamIndices->isEmpty());
    lifetimeDependenceString +=
        getSourceString(inheritLifetimeParamIndices, "copy ");
  }
  if (scopeLifetimeParamIndices) {
    assert(!scopeLifetimeParamIndices->isEmpty());
    if (inheritLifetimeParamIndices) {
      lifetimeDependenceString += ", ";
    }
    lifetimeDependenceString +=
        getSourceString(scopeLifetimeParamIndices, "borrow ");
  }
  if (isImmortal()) {
    lifetimeDependenceString += "immortal";
  }
  lifetimeDependenceString += ") ";
  return lifetimeDependenceString;
}

void LifetimeDependenceInfo::Profile(llvm::FoldingSetNodeID &ID) const {
  ID.AddBoolean(addressableParamIndicesAndImmortal.getInt());
  ID.AddInteger(targetIndex);
  if (inheritLifetimeParamIndices) {
    ID.AddInteger((uint8_t)LifetimeDependenceKind::Inherit);
    inheritLifetimeParamIndices->Profile(ID);
  }
  if (scopeLifetimeParamIndices) {
    ID.AddInteger((uint8_t)LifetimeDependenceKind::Scope);
    scopeLifetimeParamIndices->Profile(ID);
  }
  if (addressableParamIndicesAndImmortal.getPointer()) {
    ID.AddBoolean(true);
    addressableParamIndicesAndImmortal.getPointer()->Profile(ID);
  } else {
    ID.AddBoolean(false);  
  }
}

static LifetimeDependenceKind
getLifetimeDependenceKindFromType(Type sourceType) {
  if (sourceType->isEscapable()) {
    return LifetimeDependenceKind::Scope;
  }
  return LifetimeDependenceKind::Inherit;
}

// Warning: this is incorrect for Setter 'newValue' parameters. It should only
// be called for a Setter's 'self'.
static ValueOwnership getLoweredOwnership(AbstractFunctionDecl *afd) {
  if (isa<ConstructorDecl>(afd)) {
    return ValueOwnership::Owned;
  }
  if (auto *ad = dyn_cast<AccessorDecl>(afd)) {
    if (ad->getAccessorKind() == AccessorKind::Set ||
        isYieldingMutableAccessor(ad->getAccessorKind())) {
      return ValueOwnership::InOut;
    }
  }
  return ValueOwnership::Shared;
}

static bool isBitwiseCopyable(Type type, ASTContext &ctx) {
  auto *bitwiseCopyableProtocol =
      ctx.getProtocol(KnownProtocolKind::BitwiseCopyable);
  if (!bitwiseCopyableProtocol) {
    return false;
  }
  return (bool)checkConformance(type, bitwiseCopyableProtocol);
}

void LifetimeDependenceInfo::getConcatenatedData(
    SmallVectorImpl<bool> &concatenatedData) const {
  auto pushData = [&](IndexSubset *paramIndices) {
    if (paramIndices == nullptr) {
      return;
    }
    assert(!paramIndices->isEmpty());

    for (unsigned i = 0; i < paramIndices->getCapacity(); i++) {
      if (paramIndices->contains(i)) {
        concatenatedData.push_back(true);
        continue;
      }
      concatenatedData.push_back(false);
    }
  };
  if (hasInheritLifetimeParamIndices()) {
    pushData(inheritLifetimeParamIndices);
  }
  if (hasScopeLifetimeParamIndices()) {
    pushData(scopeLifetimeParamIndices);
  }
  if (hasAddressableParamIndices()) {
    pushData(addressableParamIndicesAndImmortal.getPointer());  
  }
}

class LifetimeDependenceChecker {
  AbstractFunctionDecl *afd;

  ASTContext &ctx;
  DiagnosticEngine &diags;
  SourceLoc returnLoc;

public:
  LifetimeDependenceChecker(AbstractFunctionDecl *afd):
    afd(afd), ctx(afd->getDeclContext()->getASTContext()), diags(ctx.Diags) {
    auto resultTypeRepr = afd->getResultTypeRepr();
    returnLoc = resultTypeRepr ? resultTypeRepr->getLoc() : afd->getLoc();
  }

  std::optional<LifetimeDependenceInfo> checkFuncDecl() {
    assert(isa<FuncDecl>(afd) || isa<ConstructorDecl>(afd));

    if (afd->getAttrs().hasAttribute<LifetimeAttr>()) {
      return checkAttribute();
    }

    return inferOrDiagnose();
  }

protected:
  static bool isCompatibleWithOwnership(LifetimeDependenceKind kind,
                                        Type type, ValueOwnership ownership) {
    if (kind == LifetimeDependenceKind::Inherit) {
      return true;
    }
    // Lifetime dependence always propagates through temporary BitwiseCopyable
    // values, even if the dependence is scoped.
    if (isBitwiseCopyable(type, ctx)) {
      return true;
    }
    assert(kind == LifetimeDependenceKind::Scope);
    auto loweredOwnership = ownership != ValueOwnership::Default
      ? ownership : getLoweredOwnership(afd);

    if (loweredOwnership == ValueOwnership::InOut ||
        loweredOwnership == ValueOwnership::Shared) {
      return true;
    }
    assert(loweredOwnership == ValueOwnership::Owned);
    return false;
  }

  static LifetimeDependenceInfo
  getForIndex(unsigned targetIndex, unsigned sourceIndex,
              LifetimeDependenceKind kind) {
    unsigned capacity = afd->hasImplicitSelfDecl()
      ? (afd->getParameters()->size() + 1)
      : afd->getParameters()->size();
    auto indexSubset = IndexSubset::get(ctx, capacity, {sourceIndex});
    if (kind == LifetimeDependenceKind::Scope) {
      return LifetimeDependenceInfo{/*inheritLifetimeParamIndices*/ nullptr,
        /*scopeLifetimeParamIndices*/ indexSubset,
        targetIndex,
        /*isImmortal*/ false};
    }
    return LifetimeDependenceInfo{/*inheritLifetimeParamIndices*/ indexSubset,
      /*scopeLifetimeParamIndices*/ nullptr,
      targetIndex,
      /*isImmortal*/ false};
  }

  Type getResultOrYield() {
    if (auto *accessor = dyn_cast<AccessorDecl>(afd)) {
      if (accessor->isCoroutine()) {
        auto yieldTyInContext = accessor->mapTypeIntoContext(
          accessor->getStorage()->getValueInterfaceType());
        return yieldTyInContext;
      }
    }
    Type resultType;
    if (auto fn = dyn_cast<FuncDecl>(afd)) {
      resultType = fn->getResultInterfaceType();
    } else {
      auto ctor = cast<ConstructorDecl>(afd);
      resultType = ctor->getResultInterfaceType();
    }
    return afd->mapTypeIntoContext(resultType);
  }

  std::optional<LifetimeDependenceKind>
  getDependenceKindFromDescriptor(LifetimeDescriptor descriptor,
                                  ParamDecl *decl) {
    auto loc = descriptor.getLoc();

    auto ownership = decl->getValueOwnership();
    auto type = decl->getTypeInContext();

    // For @lifetime attribute, we check if we had a "borrow" modifier, if not
    // we infer inherit dependence.
    auto parsedLifetimeKind = descriptor.getParsedLifetimeDependenceKind();
    if (parsedLifetimeKind == ParsedLifetimeDependenceKind::Scope) {
      bool isCompatible = isCompatibleWithOwnership(
        LifetimeDependenceKind::Scope, type, ownership, afd);
      if (!isCompatible) {
        diags.diagnose(
          loc, diag::lifetime_dependence_cannot_use_parsed_borrow_consuming);
        return std::nullopt;
      }
      return LifetimeDependenceKind::Scope;
    }
    if (type->isEscapable()) {
      diags.diagnose(loc,
                     diag::lifetime_dependence_invalid_inherit_escapable_type);
      return std::nullopt;
    }
    return LifetimeDependenceKind::Inherit;
  }

  // Finds the ParamDecl* and its index from a LifetimeDescriptor
  std::optional<std::pair<ParamDecl *, unsigned>>
  getParamDeclFromDescriptor(LifetimeDescriptor descriptor) {
    switch (descriptor.getDescriptorKind()) {
    case LifetimeDescriptor::DescriptorKind::Named: {
      unsigned paramIndex = 0;
      ParamDecl *candidateParam = nullptr;
      for (auto *param : *afd->getParameters()) {
        if (param->getParameterName() == descriptor.getName()) {
          candidateParam = param;
          break;
        }
        paramIndex++;
      }
      if (!candidateParam) {
        diags.diagnose(descriptor.getLoc(),
                       diag::lifetime_dependence_invalid_param_name,
                       descriptor.getName());
        return std::nullopt;
      }
      return std::make_pair(candidateParam, paramIndex);
    }
    case LifetimeDescriptor::DescriptorKind::Ordered: {
      auto paramIndex = descriptor.getIndex();
      if (paramIndex >= afd->getParameters()->size()) {
        diags.diagnose(descriptor.getLoc(),
                       diag::lifetime_dependence_invalid_param_index,
                       paramIndex);
        return std::nullopt;
      }
      auto candidateParam = afd->getParameters()->get(paramIndex);
      return std::make_pair(candidateParam, paramIndex);
    }
    case LifetimeDescriptor::DescriptorKind::Self: {
      if (!afd->hasImplicitSelfDecl()) {
        diags.diagnose(descriptor.getLoc(),
                       diag::lifetime_dependence_invalid_self_in_static);
        return std::nullopt;
      }
      if (isa<ConstructorDecl>(afd)) {
        diags.diagnose(descriptor.getLoc(),
                       diag::lifetime_dependence_invalid_self_in_init);
        return std::nullopt;
      }
      auto *selfDecl = afd->getImplicitSelfDecl();
      return std::make_pair(selfDecl, afd->getParameters()->size());
    }
    }
  }

  std::optional<ArrayRef<LifetimeDependenceInfo>> checkAttribute() {
    SmallVector<LifetimeDependenceInfo, 1> lifetimeDependencies;
    llvm::SmallSet<unsigned, 1> lifetimeDependentTargets;
    auto lifetimeAttrs = afd->getAttrs().getAttributes<LifetimeAttr>();
    for (auto attr : lifetimeAttrs) {
      auto lifetimeDependenceInfo =
        checkAttributeEntry(attr->getLifetimeEntry());
      if (!lifetimeDependenceInfo.has_value()) {
        return std::nullopt;
      }
      auto targetIndex = lifetimeDependenceInfo->getTargetIndex();
      if (lifetimeDependentTargets.contains(targetIndex)) {
        // TODO: Diagnose at the source location of the @lifetime attribute with
        // duplicate target.
        diags.diagnose(afd->getLoc(),
                       diag::lifetime_dependence_duplicate_target);
      }
      lifetimeDependentTargets.insert(targetIndex);
      lifetimeDependencies.push_back(*lifetimeDependenceInfo);
    }

    return afd->getASTContext().AllocateCopy(lifetimeDependencies);
  }

  std::optional<LifetimeDependenceInfo>
  checkAttributeEntry(LifetimeEntry *entry) {
    auto capacity = afd->hasImplicitSelfDecl()
      ? (afd->getParameters()->size() + 1)
      : afd->getParameters()->size();

    SmallBitVector inheritIndices(capacity);
    SmallBitVector scopeIndices(capacity);

    auto updateLifetimeIndices = [&](LifetimeDescriptor descriptor,
                                     unsigned paramIndexToSet,
                                     LifetimeDependenceKind lifetimeKind) {
      if (inheritIndices.test(paramIndexToSet) ||
          scopeIndices.test(paramIndexToSet)) {
        diags.diagnose(descriptor.getLoc(),
                       diag::lifetime_dependence_duplicate_param_id);
        return true;
      }
      if (lifetimeKind == LifetimeDependenceKind::Inherit) {
        inheritIndices.set(paramIndexToSet);
      } else {
        assert(lifetimeKind == LifetimeDependenceKind::Scope);
        scopeIndices.set(paramIndexToSet);
      }
      return false;
    };

    auto targetDescriptor = entry->getTargetDescriptor();
    unsigned targetIndex;
    if (targetDescriptor.has_value()) {
      auto targetDeclAndIndex =
        getParamDeclFromDescriptor(afd, *targetDescriptor);
      if (!targetDeclAndIndex.has_value()) {
        return std::nullopt;
      }
      // TODO: support dependencies on non-inout parameters.
      if (!targetDeclAndIndex->first->isInOut()) {
        diags.diagnose(targetDeclAndIndex->first,
                       diag::lifetime_parameter_requires_inout);
      }
      targetIndex = targetDeclAndIndex->second;
    } else {
      targetIndex = afd->hasImplicitSelfDecl()
        ? afd->getParameters()->size() + 1
        : afd->getParameters()->size();
    }

    for (auto source : entry->getSources()) {
      if (source.isImmortal()) {
        auto immortalParam =
          std::find_if(afd->getParameters()->begin(),
                       afd->getParameters()->end(), [](ParamDecl *param) {
                         return strcmp(param->getName().get(), "immortal") == 0;
                       });
        if (immortalParam != afd->getParameters()->end()) {
          diags.diagnose(*immortalParam,
                         diag::lifetime_dependence_immortal_conflict_name);
          return std::nullopt;
        }
        return LifetimeDependenceInfo(nullptr, nullptr, targetIndex,
                                      /*isImmortal*/ true);
      }

      auto paramDeclAndIndex = getParamDeclFromDescriptor(afd, source);
      if (!paramDeclAndIndex.has_value()) {
        return std::nullopt;
      }
      auto lifetimeKind =
        getLifetimeDependenceKind(source, afd, paramDeclAndIndex->first);
      if (!lifetimeKind.has_value()) {
        return std::nullopt;
      }
      bool hasError =
        updateLifetimeIndices(source, paramDeclAndIndex->second, *lifetimeKind);
      if (hasError) {
        return std::nullopt;
      }
    }

    return LifetimeDependenceInfo(
      inheritIndices.any() ? IndexSubset::get(ctx, inheritIndices) : nullptr,
      scopeIndices.any() ? IndexSubset::get(ctx, scopeIndices) : nullptr,
      targetIndex, /*isImmortal*/ false);
  }

  // Return nullopt if no inference is needed (e.g. for Escapable results). If
  // inference is needed but not satisfied, diagnose the error. Otherwise return
  // the inferred dependencies.
  std::optional<LifetimeDependenceInfo> inferOrDiagnose() {
    auto resultType = getResultOrYield(afd);
    if (resultType->hasError()) {
      return std::nullopt;
    }

    // FIXME: This check is temporary until rdar://139976667 is fixed.
    // ModuleType created with ModuleType::get methods are ~Copyable and
    // ~Escapable because the Copyable and Escapable conformance is not added to
    // them by default.
    if (resultType->is<ModuleType>()) {
      return std::nullopt;
    }

    // Methods and functions that return a non-Escapable value.
    if (!resultType->isEscapable()) {
      // non-Escapable results require the LifetimeDependence feature.
      if (!ctx.LangOpts.hasFeature(Feature::LifetimeDependence)) {
        diags.diagnose(returnLoc, diag::lifetime_dependence_feature_required);
        return std::nullopt;
      }
      if (!cd && afd->hasImplicitSelfDecl()) {
        return inferNonEscapableResultOnSelf();
      }
      return inferNonEscapableResultOnParam();
    }
    if (!ctx.LangOpts.EnableExperimentalLifetimeDependenceInference) {
      return std::nullopt;
    }

    // Mutating methods
    if (!cd && afd->hasImplicitSelfDecl()
        && afd->getImplicitSelfDecl()->isInOut()) {
      return inferMutatingSelfOnParam(afd);
    }

    // Setters
    if (auto accessor = dyn_cast<AccessorDecl>(afd)) {
      if (accessor->getAccessorKind() == AccessorKind::Set) {
        return inferSetterSelfOnParam(accessor);
      }
    }

    return std::nullopt;
  }

  // Return nullopt if inference cannot find a candidate.
  std::optional<LifetimeDependenceInfo> inferNonEscapableResultOnSelf() {
    Type selfTypeInContext = dc->getSelfTypeInContext();

    if (!ctx.LangOpts.EnableExperimentalLifetimeDependenceInference) {
      if (afd->getParameters()->size() > 0) {
        //!!! handle implicit init??
        diags.diagnose(
          returnLoc,
          diag::lifetime_dependence_cannot_infer_ambiguous_candidate,
          "");
        return std::nullopt;
      }
      // Do not infer non-escapable dependence kind -- it is ambiguous.
      if (!selfTypeInContext->isEscapable()) {
        //!!! handle implicit init??
        diags.diagnose(
          returnLoc,
          diag::lifetime_dependence_cannot_infer_no_kind,
          "self");
        return std::nullopt;
      }
      return LifetimeDependenceInfo::getForIndex(afd, resultIndex, selfIndex, kind);
    }
    if (selfTypeInContext->isEscapable()
        && isBitwiseCopyable(selfTypeInContext, ctx)) {
      diags.diagnose(
        returnLoc,
        diag::lifetime_dependence_method_escapable_bitwisecopyable_self);
      return std::nullopt;
    }
    auto kind = getLifetimeDependenceKindFromType(selfTypeInContext);
    auto selfOwnership = afd->getImplicitSelfDecl()->getValueOwnership();
    if (!isCompatibleWithOwnership(kind, selfTypeInContext, selfOwnership)) {
      diags.diagnose(returnLoc,
                     diag::lifetime_dependence_invalid_self_ownership);
      return std::nullopt;
    }
    // Infer method dependence: result depends on self.
    //
    // This includes _modify. A _modify's yielded value depends on self. The
    // caller of the _modify ensures that the 'self' depends on any value stored
    // to the yielded address.
    unsigned selfIndex = afd->getParameters()->size();
    unsigned resultIndex = selfIndex + 1;
    return
      LifetimeDependenceInfo::getForIndex(afd, resultIndex, selfIndex, kind);
  }

  std::optional<LifetimeDependenceInfo> inferNonEscapableResultOnParam() {
    assert(!afd->hasImplicitSelfDecl());
    unsigned resultIndex = afd->getParameters()->size();

    // --- empty types don't need a dependence source

    auto *cd = dyn_cast<ConstructorDecl>(afd);
    //!!! does a constructordecl always or never have implicit self???

    // Allow empty types to be initialized by default without any dependencies.
    if (cd && cd->getParameters()->size() == 0) {
      if (cd->isImplicit()) {
        return std::nullopt;
      }
      if (auto *sf = afd->getParentSourceFile()) {
        // The AST printer makes implicit initializers explicit, but does not
        // print the @lifetime annotations. Until that is fixed, avoid
        // diagnosing this as an error.
        if (sf->Kind == SourceFileKind::SIL) {
          return std::nullopt;
        }
      }
    }

    // A single escapable parameter is an unambiguous borrow dependence.
    if (!ctx.LangOpts.EnableExperimentalLifetimeDependenceInference) {
      if (afd->getParameters()->size() > 1) {
        //!!! handle implicit init??
        diags.diagnose(
          returnLoc,
          diag::lifetime_dependence_cannot_infer_ambiguous_candidate,
          "");
        return std::nullopt;
      }
      // Do not infer non-escapable dependence kind -- it is ambiguous.
      auto *param = afd->getParameters()[0];
      Type paramTypeInContext =
        afd->mapTypeIntoContext(param->getInterfaceType());
      if (!paramTypeInContext->isEscapable()) {
        //!!! handle implicit init??
        diags.diagnose(
          returnLoc,
          diag::lifetime_dependence_cannot_infer_no_kind,
          "self");
        return std::nullopt;
      }
      return LifetimeDependenceInfo::getForIndex(
        afd, resultIndex, 0, LifetimeDependenceKind::Scope);
    }

    std::optional<unsigned> candidateParamIndex;
    std::optional<LifetimeDependenceKind> candidateLifetimeKind;
    unsigned paramIndex = 0;
    bool hasParamError = false;
    for (auto *param : *afd->getParameters()) {
      SWIFT_DEFER { paramIndex++; };
      Type paramTypeInContext =
        afd->mapTypeIntoContext(param->getInterfaceType());
      if (paramTypeInContext->hasError()) {
        hasParamError = true;
        continue;
      }
      auto paramOwnership = param->getValueOwnership();
      if (paramTypeInContext->isEscapable()) {
        if (isBitwiseCopyable(paramTypeInContext, ctx)) {
          continue;
        }
        if (paramOwnership == ValueOwnership::Default) {
          continue;
        }
      }

      candidateLifetimeKind =
        getLifetimeDependenceKindFromType(paramTypeInContext);
      if (!isCompatibleWithOwnership(
            *candidateLifetimeKind, paramTypeInContext, paramOwnership)) {
        continue;
      }
      if (candidateParamIndex) {
        if (cd && afd->isImplicit()) {
          diags.diagnose(
            returnLoc,
            diag::lifetime_dependence_cannot_infer_ambiguous_candidate,
            "on implicit initializer");
          return std::nullopt;
        }
        diags.diagnose(
          returnLoc,
          diag::lifetime_dependence_cannot_infer_ambiguous_candidate,
          "");
        return std::nullopt;
      }
      candidateParamIndex = paramIndex;
    }

    if (!candidateParamIndex && !hasParamError) {
      if (cd && afd->isImplicit()) {
        diags.diagnose(returnLoc,
                       diag::lifetime_dependence_cannot_infer_no_candidates,
                       " on implicit initializer");
        return std::nullopt;
      }
      diags.diagnose(returnLoc,
                     diag::lifetime_dependence_cannot_infer_no_candidates, "");
      return std::nullopt;
    }
    return LifetimeDependenceInfo::getForIndex(
      afd, resultIndex, *candidateParamIndex, *candidateLifetimeKind);
  }

  /// Infer LifetimeDependenceInfo on a mutating method where 'self' is
  /// nonescapable and the result is 'void'.
  std::optional<LifetimeDependenceInfo> inferMutatingSelfOnParam() {
    Type selfTypeInContext = dc->getSelfTypeInContext();
    if (selfTypeInContext->isEscapable()) {
      return std::nullopt;
    }
    std::optional<LifetimeDependenceInfo> dep;
    for (unsigned paramIndex : range(afd->getParameters()->size())) {
      auto *param = afd->getParameters()->get(paramIndex);
      Type paramTypeInContext =
        afd->mapTypeIntoContext(param->getInterfaceType());
      if (paramTypeInContext->hasError()) {
        continue;
      }
      if (paramTypeInContext->isEscapable()) {
        continue;
      }
      if (dep) {
        // Don't infer dependence on multiple nonescapable parameters. We may
        // want to do this in the future if dependsOn(self: arg1, arg2) syntax
        // is too cumbersome.
        return std::nullopt;
      }
      int selfIndex = afd->getParameters()->size();
      dep = LifetimeDependenceInfo::getForIndex(
        afd, selfIndex, paramIndex, LifetimeDependenceKind::Inherit);
    }
    return dep;
  }

  /// Infer LifetimeDependence on a setter where 'self' is nonescapable.
  std::optional<LifetimeDependenceInfo> inferSetterSelfOnParam() {
    auto *param = afd->getParameters()->get(0);
    Type paramTypeInContext =
      afd->mapTypeIntoContext(param->getInterfaceType());
    if (paramTypeInContext->hasError()) {
      return std::nullopt;
    }
    if (paramTypeInContext->isEscapable()) {
      return std::nullopt;
    }
    auto kind = getLifetimeDependenceKindFromType(paramTypeInContext);
    return LifetimeDependenceInfo::getForIndex(
      afd, /*selfIndex */ afd->getParameters()->size(), 0,
      LifetimeDependenceInfo::Inherit);
  }
};

std::optional<llvm::ArrayRef<LifetimeDependenceInfo>>
LifetimeDependenceInfo::get(AbstractFunctionDecl *afd) {
  auto resultDependence = LifetimeDependenceChecker(afd).checkFuncDecl()
  if (!resultDependence.has_value()) {
    return std::nullopt;
  }
  SmallVector<LifetimeDependenceInfo, 1> lifetimeDependencies;
  lifetimeDependencies.push_back(*resultDependence);
  return afd->getASTContext().AllocateCopy(lifetimeDependencies);
}

// This implements the logic for SIL type descriptors similar to source-level
// logic in LifetimeDependenceChecker::checkAttributeEntry(). The SIL context is
// substantially different from Sema.
static std::optional<LifetimeDependenceInfo> checkSILTypeModifiers(
  LifetimeDependentTypeRepr *lifetimeDependentRepr, unsigned targetIndex,
  ArrayRef<SILParameterInfo> params, DeclContext *dc) {
  auto capacity = params.size(); // SIL parameters include self

  SmallBitVector inheritLifetimeParamIndices(capacity);
  SmallBitVector scopeLifetimeParamIndices(capacity);
  SmallBitVector addressableLifetimeParamIndices(capacity);

  auto updateLifetimeDependenceInfo =
    [&](LifetimeDescriptor descriptor,
        unsigned paramIndexToSet,
        ParameterConvention paramConvention) {
      auto loc = descriptor.getLoc();
      auto kind = descriptor.getParsedLifetimeDependenceKind();

      if (kind == ParsedLifetimeDependenceKind::Scope &&
          isConsumedParameterInCallee(paramConvention)) {
        diags.diagnose(loc, diag::lifetime_dependence_cannot_use_kind, "_scope",
                       getStringForParameterConvention(paramConvention));
        return true;
      }

      if (inheritLifetimeParamIndices.test(paramIndexToSet) ||
          scopeLifetimeParamIndices.test(paramIndexToSet)) {
        diags.diagnose(loc, diag::lifetime_dependence_duplicate_param_id);
        return true;
      }
      if (kind == ParsedLifetimeDependenceKind::Inherit) {
        inheritLifetimeParamIndices.set(paramIndexToSet);
      } else {
        assert(kind == ParsedLifetimeDependenceKind::Scope);
        scopeLifetimeParamIndices.set(paramIndexToSet);
      }
      return false;
    };

  for (auto source : lifetimeDependentRepr->getLifetimeEntry()->getSources())
  {
    switch (source.getDescriptorKind()) {
    case LifetimeDescriptor::DescriptorKind::Ordered: {
      auto index = source.getIndex();
      if (index > capacity) {
        diags.diagnose(source.getLoc(),
                       diag::lifetime_dependence_invalid_param_index, index);
        return std::nullopt;
      }
      auto param = params[index];
      auto paramConvention = param.getConvention();
      if (updateLifetimeDependenceInfo(source, index, paramConvention)) {
        return std::nullopt;
      }
      if (source.isAddressable()) {
        addressableLifetimeParamIndices.set(index);
      }
      break;
    }
    case LifetimeDescriptor::DescriptorKind::Named: {
      assert(source.isImmortal());
      return LifetimeDependenceInfo(/*inheritLifetimeParamIndices*/ nullptr,
                                    /*scopeLifetimeParamIndices*/ nullptr,
                                    targetIndex,
                                    /*isImmortal*/ true);
    }
    default:
      llvm_unreachable("SIL can only have ordered or immortal lifetime "
                       "dependence specifier kind");
    }
  }

  return LifetimeDependenceInfo(
    inheritLifetimeParamIndices.any()
    ? IndexSubset::get(ctx, inheritLifetimeParamIndices)
    : nullptr,
    scopeLifetimeParamIndices.any()
    ? IndexSubset::get(ctx, scopeLifetimeParamIndices)
    : nullptr,
    targetIndex,
    /*isImmortal*/ false,
    addressableLifetimeParamIndices.any()
    ? IndexSubset::get(ctx, addressableLifetimeParamIndices)
    : nullptr);
}

std::optional<llvm::ArrayRef<LifetimeDependenceInfo>>
LifetimeDependenceInfo::getFromSIL(FunctionTypeRepr *funcRepr,
                                   ArrayRef<SILParameterInfo> params,
                                   ArrayRef<SILResultInfo> results,
                                   DeclContext *dc) {
  SmallVector<LifetimeDependenceInfo, 1> lifetimeDependencies;

  auto getLifetimeDependenceFromTypeModifiers =
      [&](TypeRepr *typeRepr,
          unsigned targetIndex) -> std::optional<LifetimeDependenceInfo> {
    auto *lifetimeTypeRepr =
        dyn_cast_or_null<LifetimeDependentTypeRepr>(typeRepr);
    if (!lifetimeTypeRepr) {
      return std::nullopt;
    }
    return checkSILTypeModifiers(lifetimeTypeRepr, targetIndex, params, dc);
  };

  auto argsTypeRepr = funcRepr->getArgsTypeRepr()->getElements();
  for (unsigned targetIndex : indices(argsTypeRepr)) {
    if (auto result = getLifetimeDependenceFromTypeModifiers(
            argsTypeRepr[targetIndex].Type, targetIndex)) {
      lifetimeDependencies.push_back(*result);
    }
  }

  auto result = getLifetimeDependenceFromTypeModifiers(
    funcRepr->getResultTypeRepr(), params.size(), dc);
  if (result) {
    lifetimeDependencies.push_back(*result);
  }

  return dc->getASTContext().AllocateCopy(lifetimeDependencies);
}

void LifetimeDependenceInfo::dump() const {
  llvm::errs() << "target: " << getTargetIndex() << '\n';
  if (isImmortal()) {
    llvm::errs() << "  immortal\n";
  }
  if (auto scoped = getScopeIndices()) {
    llvm::errs() << "  scoped: ";
    scoped->dump();
  }
  if (auto inherited = getInheritIndices()) {
    llvm::errs() << "  inherited: ";
    inherited->dump();
  }
  if (auto addressable = getAddressableIndices()) {
    llvm::errs() << "  addressable: ";
    addressable->dump();
  }
}

} // namespace swift
