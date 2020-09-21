function _createForOfIteratorHelper(o, allowArrayLike) { var it; if (typeof Symbol === "undefined" || o[Symbol.iterator] == null) { if (Array.isArray(o) || (it = _unsupportedIterableToArray(o)) || allowArrayLike && o && typeof o.length === "number") { if (it) o = it; var i = 0; var F = function F() {}; return { s: F, n: function n() { if (i >= o.length) return { done: true }; return { done: false, value: o[i++] }; }, e: function e(_e) { throw _e; }, f: F }; } throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); } var normalCompletion = true, didErr = false, err; return { s: function s() { it = o[Symbol.iterator](); }, n: function n() { var step = it.next(); normalCompletion = step.done; return step; }, e: function e(_e2) { didErr = true; err = _e2; }, f: function f() { try { if (!normalCompletion && it["return"] != null) it["return"](); } finally { if (didErr) throw err; } } }; }

function _unsupportedIterableToArray(o, minLen) { if (!o) return; if (typeof o === "string") return _arrayLikeToArray(o, minLen); var n = Object.prototype.toString.call(o).slice(8, -1); if (n === "Object" && o.constructor) n = o.constructor.name; if (n === "Map" || n === "Set") return Array.from(o); if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen); }

function _arrayLikeToArray(arr, len) { if (len == null || len > arr.length) len = arr.length; for (var i = 0, arr2 = new Array(len); i < len; i++) { arr2[i] = arr[i]; } return arr2; }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }

function _createClass(Constructor, protoProps, staticProps) { if (protoProps) _defineProperties(Constructor.prototype, protoProps); if (staticProps) _defineProperties(Constructor, staticProps); return Constructor; }

(window["webpackJsonp"] = window["webpackJsonp"] || []).push([["default~layouts-back-office-layout-back-office-layout-module~layouts-back-office-mini-layout-back-of~325f412c"], {
  /***/
  "./node_modules/raw-loader/dist/cjs.js!./src/gaxon/components/navigation/horizontal/nav-collapse/nav-collapse.component.html":
  /*!***********************************************************************************************************************************!*\
    !*** ./node_modules/raw-loader/dist/cjs.js!./src/gaxon/components/navigation/horizontal/nav-collapse/nav-collapse.component.html ***!
    \***********************************************************************************************************************************/

  /*! exports provided: default */

  /***/
  function node_modulesRawLoaderDistCjsJsSrcGaxonComponentsNavigationHorizontalNavCollapseNavCollapseComponentHtml(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony default export */


    __webpack_exports__["default"] = "<a class=\"nav-link dt-side-nav__arrow\" (click)=\"toggleOpen($event)\" [ngClass]=\"{active:isOpen}\">\r\n  <ng-container *ngTemplateOutlet=\"itemContent\"></ng-container>\r\n</a>\r\n\r\n<ng-template #itemContent>\r\n  <gx-icon class=\"nav-link-icon\" *ngIf=\"item.icon\" name=\"{{item.icon}}\" size=\"xl\" [fw]=\"true\"></gx-icon>\r\n  <span *ngIf=\"item.translate\" class=\"nav-text\" [translate]=\"item.translate\"></span>\r\n  <span *ngIf=\"!item.translate\" class=\"nav-text\">{{item.title}}</span>\r\n</ng-template>\r\n\r\n<div class=\"sub-menu\">\r\n  <ng-container *ngFor=\"let item of item.children\">\r\n    <app-nav-horizontal-collapse *ngIf=\"item.type=='group'\" [item]=\"item\"></app-nav-horizontal-collapse>\r\n    <app-nav-horizontal-collapse *ngIf=\"item.type=='collapse'\" [item]=\"item\"></app-nav-horizontal-collapse>\r\n    <app-nav-horizontal-item *ngIf=\"item.type=='item'\" [item]=\"item\"></app-nav-horizontal-item>\r\n  </ng-container>\r\n</div>\r\n";
    /***/
  },

  /***/
  "./node_modules/raw-loader/dist/cjs.js!./src/gaxon/components/navigation/horizontal/nav-item/nav-item.component.html":
  /*!***************************************************************************************************************************!*\
    !*** ./node_modules/raw-loader/dist/cjs.js!./src/gaxon/components/navigation/horizontal/nav-item/nav-item.component.html ***!
    \***************************************************************************************************************************/

  /*! exports provided: default */

  /***/
  function node_modulesRawLoaderDistCjsJsSrcGaxonComponentsNavigationHorizontalNavItemNavItemComponentHtml(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony default export */


    __webpack_exports__["default"] = "<a class=\"nav-link\" *ngIf=\"item.url\" [routerLink]=\"[item.url]\" routerLinkActive=\"active\"\r\n   [routerLinkActiveOptions]=\"{exact: item.exactMatch || false}\" matRipple>\r\n  <ng-container *ngTemplateOutlet=\"itemTemplate\"></ng-container>\r\n</a>\r\n\r\n<span class=\"nav-link\" *ngIf=\"item.function\" (click)=\"item.function()\">\r\n  <ng-container *ngTemplateOutlet=\"itemTemplate\"></ng-container>\r\n</span>\r\n\r\n<ng-template #itemTemplate>\r\n  <gx-icon *ngIf=\"item.icon\" name=\"{{item.icon}}\" size=\"xl\" [fw]=\"true\"></gx-icon>\r\n  <span *ngIf=\"item.translate\" class=\"nav-text\" [translate]=\"item.translate\"></span>\r\n  <span *ngIf=\"!item.translate\" class=\"nav-text\">{{item.title}}</span>\r\n</ng-template>\r\n";
    /***/
  },

  /***/
  "./node_modules/raw-loader/dist/cjs.js!./src/gaxon/components/navigation/horizontal/nav-mega/nav-mega.component.html":
  /*!***************************************************************************************************************************!*\
    !*** ./node_modules/raw-loader/dist/cjs.js!./src/gaxon/components/navigation/horizontal/nav-mega/nav-mega.component.html ***!
    \***************************************************************************************************************************/

  /*! exports provided: default */

  /***/
  function node_modulesRawLoaderDistCjsJsSrcGaxonComponentsNavigationHorizontalNavMegaNavMegaComponentHtml(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony default export */


    __webpack_exports__["default"] = "<a class=\"nav-link dt-side-nav__arrow\" (click)=\"toggleOpen($event)\" [ngClass]=\"{active:isOpen}\">\r\n  <ng-container *ngTemplateOutlet=\"itemContent\"></ng-container>\r\n</a>\r\n\r\n<ng-template #itemContent>\r\n  <gx-icon class=\"nav-link-icon\" *ngIf=\"item.icon\" name=\"{{item.icon}}\" size=\"xl\" [fw]=\"true\"></gx-icon>\r\n  <span *ngIf=\"item.translate\" class=\"nav-text\" [translate]=\"item.translate\"></span>\r\n  <span *ngIf=\"!item.translate\" class=\"nav-text\">{{item.title}}</span>\r\n</ng-template>\r\n\r\n<div class=\"sub-menu\">\r\n  <ng-container *ngFor=\"let item of item.children\">\r\n    <ng-container *ngIf=\"item.type=='collapse'\">\r\n      <div class=\"nav-item\">\r\n        <!-- Heading -->\r\n        <h3 *ngIf=\"item.translate || item.title\" class=\"menu-heading\">\r\n          <gx-icon class=\"nav-link-icon\" *ngIf=\"item.icon\" name=\"{{item.icon}}\" size=\"xl\" [fw]=\"true\"></gx-icon>\r\n          <span [translate]=\"item.translate\">{{item.title}}</span>\r\n        </h3>\r\n        <!-- /heading -->\r\n\r\n        <div class=\"nav-item__inner\">\r\n          <ng-container *ngFor=\"let child of item.children\">\r\n            <app-nav-horizontal-item *ngIf=\"child.type=='item'\" [item]=\"child\"></app-nav-horizontal-item>\r\n          </ng-container>\r\n        </div>\r\n      </div>\r\n    </ng-container>\r\n\r\n    <app-nav-horizontal-item *ngIf=\"item.type=='item'\" [item]=\"item\"></app-nav-horizontal-item>\r\n  </ng-container>\r\n</div>\r\n";
    /***/
  },

  /***/
  "./node_modules/raw-loader/dist/cjs.js!./src/gaxon/components/navigation/navigation.component.html":
  /*!*********************************************************************************************************!*\
    !*** ./node_modules/raw-loader/dist/cjs.js!./src/gaxon/components/navigation/navigation.component.html ***!
    \*********************************************************************************************************/

  /*! exports provided: default */

  /***/
  function node_modulesRawLoaderDistCjsJsSrcGaxonComponentsNavigationNavigationComponentHtml(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony default export */


    __webpack_exports__["default"] = "<!-- Vertical Navigation Layout -->\r\n<ng-container *ngIf=\"layout === 'vertical'\">\r\n  <ng-container *ngFor=\"let item of navigation\">\r\n    <app-nav-vertical-group *ngIf=\"item.type=='group'\" [item]=\"item\"></app-nav-vertical-group>\r\n    <app-nav-vertical-collapse *ngIf=\"item.type=='collapse'\" [item]=\"item\"></app-nav-vertical-collapse>\r\n    <app-nav-vertical-item *ngIf=\"item.type=='item'\" [item]=\"item\"></app-nav-vertical-item>\r\n  </ng-container>\r\n</ng-container>\r\n<!-- / Vertical Navigation Layout -->\r\n\r\n<!-- Horizontal Navigation Layout -->\r\n<ng-container *ngIf=\"layout === 'horizontal'\">\r\n  <ng-container *ngFor=\"let item of headerMenu\">\r\n    <app-nav-horizontal-mega *ngIf=\"item.type=='mega'\" [item]=\"item\"></app-nav-horizontal-mega>\r\n    <app-nav-horizontal-collapse *ngIf=\"item.type=='collapse'\" [item]=\"item\"></app-nav-horizontal-collapse>\r\n    <app-nav-horizontal-item *ngIf=\"item.type=='item'\" [item]=\"item\"></app-nav-horizontal-item>\r\n  </ng-container>\r\n</ng-container>\r\n<!-- / Horizontal Navigation Layout -->\r\n";
    /***/
  },

  /***/
  "./node_modules/raw-loader/dist/cjs.js!./src/gaxon/components/navigation/vertical/nav-collapse/nav-collapse.component.html":
  /*!*********************************************************************************************************************************!*\
    !*** ./node_modules/raw-loader/dist/cjs.js!./src/gaxon/components/navigation/vertical/nav-collapse/nav-collapse.component.html ***!
    \*********************************************************************************************************************************/

  /*! exports provided: default */

  /***/
  function node_modulesRawLoaderDistCjsJsSrcGaxonComponentsNavigationVerticalNavCollapseNavCollapseComponentHtml(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony default export */


    __webpack_exports__["default"] = "<a class=\"dt-side-nav__link dt-side-nav__arrow\" (click)=\"toggleOpen($event)\">\r\n  <ng-container *ngTemplateOutlet=\"itemContent\"></ng-container>\r\n</a>\r\n\r\n<ng-template #itemContent>\r\n  <gx-icon class=\"nav-link-icon\" *ngIf=\"item.icon\" name=\"{{item.icon}}\" size=\"lg\" [fw]=\"true\"></gx-icon>\r\n  <span *ngIf=\"item.translate\" class=\"dt-side-nav__text\" [translate]=\"item.translate\"></span>\r\n  <span *ngIf=\"!item.translate\" class=\"dt-side-nav__text\">{{item.title}}</span>\r\n</ng-template>\r\n\r\n<div class=\"dt-side-nav__sub-menu\" [@collapseInOut]=\"isOpen\">\r\n  <ng-container *ngFor=\"let item of item.children\">\r\n    <app-nav-vertical-group *ngIf=\"item.type=='group'\" [item]=\"item\"></app-nav-vertical-group>\r\n    <app-nav-vertical-collapse *ngIf=\"item.type=='collapse'\" [item]=\"item\"></app-nav-vertical-collapse>\r\n    <app-nav-vertical-item *ngIf=\"item.type=='item'\" [item]=\"item\"></app-nav-vertical-item>\r\n  </ng-container>\r\n</div>\r\n";
    /***/
  },

  /***/
  "./node_modules/raw-loader/dist/cjs.js!./src/gaxon/components/navigation/vertical/nav-group/nav-group.component.html":
  /*!***************************************************************************************************************************!*\
    !*** ./node_modules/raw-loader/dist/cjs.js!./src/gaxon/components/navigation/vertical/nav-group/nav-group.component.html ***!
    \***************************************************************************************************************************/

  /*! exports provided: default */

  /***/
  function node_modulesRawLoaderDistCjsJsSrcGaxonComponentsNavigationVerticalNavGroupNavGroupComponentHtml(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony default export */


    __webpack_exports__["default"] = "<div class=\"dt-side-nav__item dt-side-nav__header\">\r\n  <span *ngIf=\"item.translate\" class=\"dt-side-nav__text\" [translate]=\"item.translate\"></span>\r\n  <span *ngIf=\"!item.translate\" class=\"dt-side-nav__text\">{{item.title}}</span>\r\n</div>\r\n\r\n<ng-container *ngFor=\"let item of item.children\">\r\n  <app-nav-vertical-group *ngIf=\"item.type=='group'\" [item]=\"item\"></app-nav-vertical-group>\r\n  <app-nav-vertical-collapse *ngIf=\"item.type=='collapse'\" [item]=\"item\"></app-nav-vertical-collapse>\r\n  <app-nav-vertical-item *ngIf=\"item.type=='item'\" [item]=\"item\"></app-nav-vertical-item>\r\n</ng-container>\r\n";
    /***/
  },

  /***/
  "./node_modules/raw-loader/dist/cjs.js!./src/gaxon/components/navigation/vertical/nav-item/nav-item.component.html":
  /*!*************************************************************************************************************************!*\
    !*** ./node_modules/raw-loader/dist/cjs.js!./src/gaxon/components/navigation/vertical/nav-item/nav-item.component.html ***!
    \*************************************************************************************************************************/

  /*! exports provided: default */

  /***/
  function node_modulesRawLoaderDistCjsJsSrcGaxonComponentsNavigationVerticalNavItemNavItemComponentHtml(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony default export */


    __webpack_exports__["default"] = "<a class=\"dt-side-nav__link\" *ngIf=\"item.url\" [routerLink]=\"[item.url]\" routerLinkActive=\"active\"\r\n   [routerLinkActiveOptions]=\"{exact: item.exactMatch || false}\">\r\n  <ng-container *ngTemplateOutlet=\"itemTemplate\"></ng-container>\r\n</a>\r\n\r\n<span class=\"dt-side-nav__link\" *ngIf=\"item.function\" (click)=\"item.function()\">\r\n  <ng-container *ngTemplateOutlet=\"itemTemplate\"></ng-container>\r\n</span>\r\n\r\n<ng-template #itemTemplate>\r\n  <gx-icon *ngIf=\"item.icon\" name=\"{{item.icon}}\" size=\"lg\" [fw]=\"true\"></gx-icon>\r\n  <span *ngIf=\"item.translate\" class=\"dt-side-nav__text\" [translate]=\"item.translate\"></span>\r\n  <span *ngIf=\"!item.translate\" class=\"dt-side-nav__text\">{{item.title}}</span>\r\n</ng-template>\r\n";
    /***/
  },

  /***/
  "./src/gaxon/components/navigation/horizontal/nav-collapse/nav-collapse.component.scss":
  /*!*********************************************************************************************!*\
    !*** ./src/gaxon/components/navigation/horizontal/nav-collapse/nav-collapse.component.scss ***!
    \*********************************************************************************************/

  /*! exports provided: default */

  /***/
  function srcGaxonComponentsNavigationHorizontalNavCollapseNavCollapseComponentScss(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony default export */


    __webpack_exports__["default"] = ":host {\n  display: block; }\n\n/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbInNyYy9nYXhvbi9jb21wb25lbnRzL25hdmlnYXRpb24vaG9yaXpvbnRhbC9uYXYtY29sbGFwc2UvQzpcXFVzZXJzXFx2aW5vXFx2aW5vdGhcXHRydXN0cHJpbWUtdjMuMVxcdHJ1c3QtcHJpbWUtdjMvc3JjXFxnYXhvblxcY29tcG9uZW50c1xcbmF2aWdhdGlvblxcaG9yaXpvbnRhbFxcbmF2LWNvbGxhcHNlXFxuYXYtY29sbGFwc2UuY29tcG9uZW50LnNjc3MiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7RUFDRSxjQUFjLEVBQUEiLCJmaWxlIjoic3JjL2dheG9uL2NvbXBvbmVudHMvbmF2aWdhdGlvbi9ob3Jpem9udGFsL25hdi1jb2xsYXBzZS9uYXYtY29sbGFwc2UuY29tcG9uZW50LnNjc3MiLCJzb3VyY2VzQ29udGVudCI6WyI6aG9zdHtcclxuICBkaXNwbGF5OiBibG9jaztcclxufVxyXG4iXX0= */";
    /***/
  },

  /***/
  "./src/gaxon/components/navigation/horizontal/nav-collapse/nav-collapse.component.ts":
  /*!*******************************************************************************************!*\
    !*** ./src/gaxon/components/navigation/horizontal/nav-collapse/nav-collapse.component.ts ***!
    \*******************************************************************************************/

  /*! exports provided: NavHorizontalCollapseComponent */

  /***/
  function srcGaxonComponentsNavigationHorizontalNavCollapseNavCollapseComponentTs(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony export (binding) */


    __webpack_require__.d(__webpack_exports__, "NavHorizontalCollapseComponent", function () {
      return NavHorizontalCollapseComponent;
    });
    /* harmony import */


    var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(
    /*! tslib */
    "./node_modules/tslib/tslib.es6.js");
    /* harmony import */


    var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(
    /*! @angular/core */
    "./node_modules/@angular/core/fesm2015/core.js");
    /* harmony import */


    var _angular_router__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(
    /*! @angular/router */
    "./node_modules/@angular/router/fesm2015/router.js");
    /* harmony import */


    var _navigation_service__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(
    /*! ./../../navigation.service */
    "./src/gaxon/components/navigation/navigation.service.ts");

    var NavHorizontalCollapseComponent = /*#__PURE__*/function () {
      function NavHorizontalCollapseComponent(navigationService, router) {
        var _this = this;

        _classCallCheck(this, NavHorizontalCollapseComponent);

        this.navigationService = navigationService;
        this.router = router;
        this.classes = 'nav-collapse nav-item';
        this.isOpen = false; // Listen for route changes

        router.events.subscribe(function (event) {
          if (event instanceof _angular_router__WEBPACK_IMPORTED_MODULE_2__["NavigationEnd"]) {
            // Check if the url can be found in
            // one of the children of this item
            if (_this.isUrlInChildren(_this.item, event.urlAfterRedirects)) {
              _this.isOpen = true;
            } else {
              _this.isOpen = false;
            }
          }
        });
      }

      _createClass(NavHorizontalCollapseComponent, [{
        key: "ngOnInit",
        value: function ngOnInit() {
          // Check if the url can be found in
          // one of the children of this item
          if (this.isUrlInChildren(this.item, this.router.url)) {
            this.isOpen = true;
          }
        }
        /**
         * Toggle open
         *
         * @param event
         */

      }, {
        key: "toggleOpen",
        value: function toggleOpen(event) {
          event.preventDefault();
          this.isOpen = !this.isOpen;
        }
        /**
         * Check if the given url can be found
         * in one of the given parent's children
         *
         * @param parent
         * @param url
         * @returns {any}
         */

      }, {
        key: "isUrlInChildren",
        value: function isUrlInChildren(parent, url) {
          if (!parent.children) {
            return false;
          }

          for (var i = 0; i < parent.children.length; i++) {
            if (parent.children[i].children) {
              if (this.isUrlInChildren(parent.children[i], url)) {
                return true;
              }
            }

            if (parent.children[i].url === url || url.includes(parent.children[i].url)) {
              return true;
            }
          }

          return false;
        }
      }]);

      return NavHorizontalCollapseComponent;
    }();

    NavHorizontalCollapseComponent.ctorParameters = function () {
      return [{
        type: _navigation_service__WEBPACK_IMPORTED_MODULE_3__["NavigationService"]
      }, {
        type: _angular_router__WEBPACK_IMPORTED_MODULE_2__["Router"]
      }];
    };

    tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["HostBinding"])('class')], NavHorizontalCollapseComponent.prototype, "classes", void 0);
    tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Input"])()], NavHorizontalCollapseComponent.prototype, "item", void 0);
    NavHorizontalCollapseComponent = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Component"])({
      selector: 'app-nav-horizontal-collapse',
      template: tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(
      /*! raw-loader!./nav-collapse.component.html */
      "./node_modules/raw-loader/dist/cjs.js!./src/gaxon/components/navigation/horizontal/nav-collapse/nav-collapse.component.html"))["default"],
      styles: [tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(
      /*! ./nav-collapse.component.scss */
      "./src/gaxon/components/navigation/horizontal/nav-collapse/nav-collapse.component.scss"))["default"]]
    })], NavHorizontalCollapseComponent);
    /***/
  },

  /***/
  "./src/gaxon/components/navigation/horizontal/nav-item/nav-item.component.scss":
  /*!*************************************************************************************!*\
    !*** ./src/gaxon/components/navigation/horizontal/nav-item/nav-item.component.scss ***!
    \*************************************************************************************/

  /*! exports provided: default */

  /***/
  function srcGaxonComponentsNavigationHorizontalNavItemNavItemComponentScss(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony default export */


    __webpack_exports__["default"] = "\n/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IiIsImZpbGUiOiJzcmMvZ2F4b24vY29tcG9uZW50cy9uYXZpZ2F0aW9uL2hvcml6b250YWwvbmF2LWl0ZW0vbmF2LWl0ZW0uY29tcG9uZW50LnNjc3MifQ== */";
    /***/
  },

  /***/
  "./src/gaxon/components/navigation/horizontal/nav-item/nav-item.component.ts":
  /*!***********************************************************************************!*\
    !*** ./src/gaxon/components/navigation/horizontal/nav-item/nav-item.component.ts ***!
    \***********************************************************************************/

  /*! exports provided: NavHorizontalItemComponent */

  /***/
  function srcGaxonComponentsNavigationHorizontalNavItemNavItemComponentTs(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony export (binding) */


    __webpack_require__.d(__webpack_exports__, "NavHorizontalItemComponent", function () {
      return NavHorizontalItemComponent;
    });
    /* harmony import */


    var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(
    /*! tslib */
    "./node_modules/tslib/tslib.es6.js");
    /* harmony import */


    var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(
    /*! @angular/core */
    "./node_modules/@angular/core/fesm2015/core.js");

    var NavHorizontalItemComponent = /*#__PURE__*/function () {
      function NavHorizontalItemComponent() {
        _classCallCheck(this, NavHorizontalItemComponent);

        this.classes = 'nav-item';
      }

      _createClass(NavHorizontalItemComponent, [{
        key: "ngOnInit",
        value: function ngOnInit() {}
      }]);

      return NavHorizontalItemComponent;
    }();

    tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["HostBinding"])('class')], NavHorizontalItemComponent.prototype, "classes", void 0);
    tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Input"])()], NavHorizontalItemComponent.prototype, "item", void 0);
    NavHorizontalItemComponent = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Component"])({
      selector: 'app-nav-horizontal-item',
      template: tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(
      /*! raw-loader!./nav-item.component.html */
      "./node_modules/raw-loader/dist/cjs.js!./src/gaxon/components/navigation/horizontal/nav-item/nav-item.component.html"))["default"],
      styles: [tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(
      /*! ./nav-item.component.scss */
      "./src/gaxon/components/navigation/horizontal/nav-item/nav-item.component.scss"))["default"]]
    })], NavHorizontalItemComponent);
    /***/
  },

  /***/
  "./src/gaxon/components/navigation/horizontal/nav-mega/nav-mega.component.scss":
  /*!*************************************************************************************!*\
    !*** ./src/gaxon/components/navigation/horizontal/nav-mega/nav-mega.component.scss ***!
    \*************************************************************************************/

  /*! exports provided: default */

  /***/
  function srcGaxonComponentsNavigationHorizontalNavMegaNavMegaComponentScss(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony default export */


    __webpack_exports__["default"] = "\n/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IiIsImZpbGUiOiJzcmMvZ2F4b24vY29tcG9uZW50cy9uYXZpZ2F0aW9uL2hvcml6b250YWwvbmF2LW1lZ2EvbmF2LW1lZ2EuY29tcG9uZW50LnNjc3MifQ== */";
    /***/
  },

  /***/
  "./src/gaxon/components/navigation/horizontal/nav-mega/nav-mega.component.ts":
  /*!***********************************************************************************!*\
    !*** ./src/gaxon/components/navigation/horizontal/nav-mega/nav-mega.component.ts ***!
    \***********************************************************************************/

  /*! exports provided: NavHorizontalMegaComponent */

  /***/
  function srcGaxonComponentsNavigationHorizontalNavMegaNavMegaComponentTs(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony export (binding) */


    __webpack_require__.d(__webpack_exports__, "NavHorizontalMegaComponent", function () {
      return NavHorizontalMegaComponent;
    });
    /* harmony import */


    var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(
    /*! tslib */
    "./node_modules/tslib/tslib.es6.js");
    /* harmony import */


    var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(
    /*! @angular/core */
    "./node_modules/@angular/core/fesm2015/core.js");
    /* harmony import */


    var _angular_router__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(
    /*! @angular/router */
    "./node_modules/@angular/router/fesm2015/router.js");
    /* harmony import */


    var _navigation_service__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(
    /*! ./../../navigation.service */
    "./src/gaxon/components/navigation/navigation.service.ts");

    var NavHorizontalMegaComponent = /*#__PURE__*/function () {
      function NavHorizontalMegaComponent(navigationService, router) {
        var _this2 = this;

        _classCallCheck(this, NavHorizontalMegaComponent);

        this.navigationService = navigationService;
        this.router = router;
        this.classes = 'nav-item mega-menu';
        this.isOpen = false; // Listen for route changes

        router.events.subscribe(function (event) {
          if (event instanceof _angular_router__WEBPACK_IMPORTED_MODULE_2__["NavigationEnd"]) {
            // Check if the url can be found in
            // one of the children of this item
            if (_this2.isUrlInChildren(_this2.item, event.urlAfterRedirects)) {
              _this2.isOpen = true;
            } else {
              _this2.isOpen = false;
            }
          }
        });
      }

      _createClass(NavHorizontalMegaComponent, [{
        key: "ngOnInit",
        value: function ngOnInit() {
          // Check if the url can be found in
          // one of the children of this item
          if (this.isUrlInChildren(this.item, this.router.url)) {
            this.isOpen = true;
          }
        }
        /**
         * Toggle open
         *
         * @param event
         */

      }, {
        key: "toggleOpen",
        value: function toggleOpen(event) {
          event.preventDefault();
          this.isOpen = !this.isOpen;
        }
        /**
         * Check if the given url can be found
         * in one of the given parent's children
         *
         * @param parent
         * @param url
         * @returns {any}
         */

      }, {
        key: "isUrlInChildren",
        value: function isUrlInChildren(parent, url) {
          if (!parent.children) {
            return false;
          }

          for (var i = 0; i < parent.children.length; i++) {
            if (parent.children[i].children) {
              if (this.isUrlInChildren(parent.children[i], url)) {
                return true;
              }
            }

            if (parent.children[i].url === url || url.includes(parent.children[i].url)) {
              return true;
            }
          }

          return false;
        }
      }]);

      return NavHorizontalMegaComponent;
    }();

    NavHorizontalMegaComponent.ctorParameters = function () {
      return [{
        type: _navigation_service__WEBPACK_IMPORTED_MODULE_3__["NavigationService"]
      }, {
        type: _angular_router__WEBPACK_IMPORTED_MODULE_2__["Router"]
      }];
    };

    tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["HostBinding"])('class')], NavHorizontalMegaComponent.prototype, "classes", void 0);
    tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["HostBinding"])('class.open')], NavHorizontalMegaComponent.prototype, "isOpen", void 0);
    tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Input"])()], NavHorizontalMegaComponent.prototype, "item", void 0);
    NavHorizontalMegaComponent = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Component"])({
      selector: 'app-nav-horizontal-mega',
      template: tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(
      /*! raw-loader!./nav-mega.component.html */
      "./node_modules/raw-loader/dist/cjs.js!./src/gaxon/components/navigation/horizontal/nav-mega/nav-mega.component.html"))["default"],
      styles: [tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(
      /*! ./nav-mega.component.scss */
      "./src/gaxon/components/navigation/horizontal/nav-mega/nav-mega.component.scss"))["default"]]
    })], NavHorizontalMegaComponent);
    /***/
  },

  /***/
  "./src/gaxon/components/navigation/navigation.component.scss":
  /*!*******************************************************************!*\
    !*** ./src/gaxon/components/navigation/navigation.component.scss ***!
    \*******************************************************************/

  /*! exports provided: default */

  /***/
  function srcGaxonComponentsNavigationNavigationComponentScss(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony default export */


    __webpack_exports__["default"] = "\n/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IiIsImZpbGUiOiJzcmMvZ2F4b24vY29tcG9uZW50cy9uYXZpZ2F0aW9uL25hdmlnYXRpb24uY29tcG9uZW50LnNjc3MifQ== */";
    /***/
  },

  /***/
  "./src/gaxon/components/navigation/navigation.component.ts":
  /*!*****************************************************************!*\
    !*** ./src/gaxon/components/navigation/navigation.component.ts ***!
    \*****************************************************************/

  /*! exports provided: NavigationComponent */

  /***/
  function srcGaxonComponentsNavigationNavigationComponentTs(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony export (binding) */


    __webpack_require__.d(__webpack_exports__, "NavigationComponent", function () {
      return NavigationComponent;
    });
    /* harmony import */


    var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(
    /*! tslib */
    "./node_modules/tslib/tslib.es6.js");
    /* harmony import */


    var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(
    /*! @angular/core */
    "./node_modules/@angular/core/fesm2015/core.js");
    /* harmony import */


    var _navigation_service__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(
    /*! ./navigation.service */
    "./src/gaxon/components/navigation/navigation.service.ts");

    var NavigationComponent = /*#__PURE__*/function () {
      function NavigationComponent(navService) {
        var _this3 = this;

        _classCallCheck(this, NavigationComponent);

        this.navService = navService;
        this.layout = 'vertical';
        this.navChangedSubscription = this.navService.onNavigationChanged.subscribe(function (newNavsData) {
          _this3.navigation = newNavsData;
        });
        this.onHeaderMenuSubscription = this.navService.onHeaderMenuChanged.subscribe(function (headerMenu) {
          _this3.headerMenu = headerMenu;
        });
      }

      _createClass(NavigationComponent, [{
        key: "ngOnDestroy",
        value: function ngOnDestroy() {
          this.navChangedSubscription.unsubscribe();
        }
      }]);

      return NavigationComponent;
    }();

    NavigationComponent.ctorParameters = function () {
      return [{
        type: _navigation_service__WEBPACK_IMPORTED_MODULE_2__["NavigationService"]
      }];
    };

    tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Input"])('layout')], NavigationComponent.prototype, "layout", void 0);
    NavigationComponent = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Component"])({
      selector: 'app-navigation',
      template: tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(
      /*! raw-loader!./navigation.component.html */
      "./node_modules/raw-loader/dist/cjs.js!./src/gaxon/components/navigation/navigation.component.html"))["default"],
      encapsulation: _angular_core__WEBPACK_IMPORTED_MODULE_1__["ViewEncapsulation"].None,
      styles: [tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(
      /*! ./navigation.component.scss */
      "./src/gaxon/components/navigation/navigation.component.scss"))["default"]]
    })], NavigationComponent);
    /***/
  },

  /***/
  "./src/gaxon/components/navigation/navigation.module.ts":
  /*!**************************************************************!*\
    !*** ./src/gaxon/components/navigation/navigation.module.ts ***!
    \**************************************************************/

  /*! exports provided: NavigationModule */

  /***/
  function srcGaxonComponentsNavigationNavigationModuleTs(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony export (binding) */


    __webpack_require__.d(__webpack_exports__, "NavigationModule", function () {
      return NavigationModule;
    });
    /* harmony import */


    var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(
    /*! tslib */
    "./node_modules/tslib/tslib.es6.js");
    /* harmony import */


    var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(
    /*! @angular/core */
    "./node_modules/@angular/core/fesm2015/core.js");
    /* harmony import */


    var _angular_common__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(
    /*! @angular/common */
    "./node_modules/@angular/common/fesm2015/common.js");
    /* harmony import */


    var _angular_router__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(
    /*! @angular/router */
    "./node_modules/@angular/router/fesm2015/router.js");
    /* harmony import */


    var _ngx_translate_core__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(
    /*! @ngx-translate/core */
    "./node_modules/@ngx-translate/core/fesm2015/ngx-translate-core.js");
    /* harmony import */


    var _gx_icon_gx_icon_module__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(
    /*! ../gx-icon/gx-icon.module */
    "./src/gaxon/components/gx-icon/gx-icon.module.ts");
    /* harmony import */


    var _vertical_nav_collapse_nav_collapse_component__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(
    /*! ./vertical/nav-collapse/nav-collapse.component */
    "./src/gaxon/components/navigation/vertical/nav-collapse/nav-collapse.component.ts");
    /* harmony import */


    var _vertical_nav_group_nav_group_component__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(
    /*! ./vertical/nav-group/nav-group.component */
    "./src/gaxon/components/navigation/vertical/nav-group/nav-group.component.ts");
    /* harmony import */


    var _vertical_nav_item_nav_item_component__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(
    /*! ./vertical/nav-item/nav-item.component */
    "./src/gaxon/components/navigation/vertical/nav-item/nav-item.component.ts");
    /* harmony import */


    var _horizontal_nav_item_nav_item_component__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(
    /*! ./horizontal/nav-item/nav-item.component */
    "./src/gaxon/components/navigation/horizontal/nav-item/nav-item.component.ts");
    /* harmony import */


    var _horizontal_nav_collapse_nav_collapse_component__WEBPACK_IMPORTED_MODULE_10__ = __webpack_require__(
    /*! ./horizontal/nav-collapse/nav-collapse.component */
    "./src/gaxon/components/navigation/horizontal/nav-collapse/nav-collapse.component.ts");
    /* harmony import */


    var _horizontal_nav_mega_nav_mega_component__WEBPACK_IMPORTED_MODULE_11__ = __webpack_require__(
    /*! ./horizontal/nav-mega/nav-mega.component */
    "./src/gaxon/components/navigation/horizontal/nav-mega/nav-mega.component.ts");
    /* harmony import */


    var _navigation_component__WEBPACK_IMPORTED_MODULE_12__ = __webpack_require__(
    /*! ./navigation.component */
    "./src/gaxon/components/navigation/navigation.component.ts");

    var NavigationModule = function NavigationModule() {
      _classCallCheck(this, NavigationModule);
    };

    NavigationModule = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["NgModule"])({
      imports: [_angular_common__WEBPACK_IMPORTED_MODULE_2__["CommonModule"], _ngx_translate_core__WEBPACK_IMPORTED_MODULE_4__["TranslateModule"], _gx_icon_gx_icon_module__WEBPACK_IMPORTED_MODULE_5__["GxIconModule"], _angular_router__WEBPACK_IMPORTED_MODULE_3__["RouterModule"]],
      declarations: [_navigation_component__WEBPACK_IMPORTED_MODULE_12__["NavigationComponent"], _vertical_nav_collapse_nav_collapse_component__WEBPACK_IMPORTED_MODULE_6__["NavVerticalCollapseComponent"], _vertical_nav_group_nav_group_component__WEBPACK_IMPORTED_MODULE_7__["NavVerticalGroupComponent"], _vertical_nav_item_nav_item_component__WEBPACK_IMPORTED_MODULE_8__["NavVerticalItemComponent"], _horizontal_nav_item_nav_item_component__WEBPACK_IMPORTED_MODULE_9__["NavHorizontalItemComponent"], _horizontal_nav_collapse_nav_collapse_component__WEBPACK_IMPORTED_MODULE_10__["NavHorizontalCollapseComponent"], _horizontal_nav_mega_nav_mega_component__WEBPACK_IMPORTED_MODULE_11__["NavHorizontalMegaComponent"]],
      exports: [_navigation_component__WEBPACK_IMPORTED_MODULE_12__["NavigationComponent"]]
    })], NavigationModule);
    /***/
  },

  /***/
  "./src/gaxon/components/navigation/vertical/nav-collapse/nav-collapse.component.scss":
  /*!*******************************************************************************************!*\
    !*** ./src/gaxon/components/navigation/vertical/nav-collapse/nav-collapse.component.scss ***!
    \*******************************************************************************************/

  /*! exports provided: default */

  /***/
  function srcGaxonComponentsNavigationVerticalNavCollapseNavCollapseComponentScss(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony default export */


    __webpack_exports__["default"] = "\n/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IiIsImZpbGUiOiJzcmMvZ2F4b24vY29tcG9uZW50cy9uYXZpZ2F0aW9uL3ZlcnRpY2FsL25hdi1jb2xsYXBzZS9uYXYtY29sbGFwc2UuY29tcG9uZW50LnNjc3MifQ== */";
    /***/
  },

  /***/
  "./src/gaxon/components/navigation/vertical/nav-collapse/nav-collapse.component.ts":
  /*!*****************************************************************************************!*\
    !*** ./src/gaxon/components/navigation/vertical/nav-collapse/nav-collapse.component.ts ***!
    \*****************************************************************************************/

  /*! exports provided: NavVerticalCollapseComponent */

  /***/
  function srcGaxonComponentsNavigationVerticalNavCollapseNavCollapseComponentTs(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony export (binding) */


    __webpack_require__.d(__webpack_exports__, "NavVerticalCollapseComponent", function () {
      return NavVerticalCollapseComponent;
    });
    /* harmony import */


    var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(
    /*! tslib */
    "./node_modules/tslib/tslib.es6.js");
    /* harmony import */


    var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(
    /*! @angular/core */
    "./node_modules/@angular/core/fesm2015/core.js");
    /* harmony import */


    var _angular_router__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(
    /*! @angular/router */
    "./node_modules/@angular/router/fesm2015/router.js");
    /* harmony import */


    var _mix_animations__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(
    /*! ../../../../mix/animations */
    "./src/gaxon/mix/animations.ts");
    /* harmony import */


    var _navigation_service__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(
    /*! ./../../navigation.service */
    "./src/gaxon/components/navigation/navigation.service.ts");

    var NavVerticalCollapseComponent = /*#__PURE__*/function () {
      function NavVerticalCollapseComponent(navigationService, router) {
        var _this4 = this;

        _classCallCheck(this, NavVerticalCollapseComponent);

        this.navigationService = navigationService;
        this.router = router;
        this.classes = 'nav-collapse dt-side-nav__item';
        this.isOpen = false;
        this.isSelected = false; // Listen for route changes

        router.events.subscribe(function (event) {
          if (event instanceof _angular_router__WEBPACK_IMPORTED_MODULE_2__["NavigationEnd"]) {
            // Check if the url can be found in
            // one of the children of this item
            if (_this4.isUrlInChildren(_this4.item, event.urlAfterRedirects)) {
              _this4.expand();

              _this4.isSelected = true;
            } else {
              _this4.collapse();

              _this4.isSelected = false;
            }
          }
        }); // Listen for collapsing of any navigation item

        this.navigationService.onNavCollapseToggled.subscribe(function (clickedItem) {
          if (clickedItem && clickedItem.children) {
            // Check if the clicked item is one
            // of the children of this item
            if (_this4.isChildrenOf(_this4.item, clickedItem)) {
              return;
            } // Check if the url can be found in
            // one of the children of this item


            if (_this4.isUrlInChildren(_this4.item, _this4.router.url)) {
              return;
            } // If the clicked item is not this item, collapse...


            if (_this4.item !== clickedItem) {
              _this4.collapse();
            }
          }
        });
      }

      _createClass(NavVerticalCollapseComponent, [{
        key: "ngOnInit",
        value: function ngOnInit() {
          // Check if the url can be found in
          // one of the children of this item
          if (this.isUrlInChildren(this.item, this.router.url)) {
            this.expand();
            this.isSelected = true;
          } else {
            this.collapse();
            this.isSelected = false;
          }
        }
        /**
         * Toggle collapse
         *
         * @param ev
         */

      }, {
        key: "toggleOpen",
        value: function toggleOpen(ev) {
          ev.preventDefault();
          this.isOpen = !this.isOpen; // Navigation collapse toggled...

          this.navigationService.onNavCollapseToggled.emit(this.item);
          this.navigationService.onNavCollapseToggle.emit();
        }
        /**
         * Expand the collapsable navigation
         */

      }, {
        key: "expand",
        value: function expand() {
          if (this.isOpen) {
            return;
          }

          this.isOpen = true;
          this.navigationService.onNavCollapseToggle.emit();
        }
        /**
         * Collapse the collapsable navigation
         */

      }, {
        key: "collapse",
        value: function collapse() {
          if (!this.isOpen) {
            return;
          }

          this.isOpen = false;
          this.navigationService.onNavCollapseToggle.emit();
        }
        /**
         * Check if the given parent has the
         * given item in one of its children
         *
         * @param parent
         * @param item
         * @return {any}
         */

      }, {
        key: "isChildrenOf",
        value: function isChildrenOf(parent, item) {
          if (!parent.children) {
            return false;
          }

          if (parent.children.indexOf(item) !== -1) {
            return true;
          }

          var _iterator = _createForOfIteratorHelper(parent.children),
              _step;

          try {
            for (_iterator.s(); !(_step = _iterator.n()).done;) {
              var children = _step.value;

              if (children.children) {
                return this.isChildrenOf(children, item);
              }
            }
          } catch (err) {
            _iterator.e(err);
          } finally {
            _iterator.f();
          }
        }
        /**
         * Check if the given url can be found
         * in one of the given parent's children
         *
         * @param parent
         * @param url
         * @returns {any}
         */

      }, {
        key: "isUrlInChildren",
        value: function isUrlInChildren(parent, url) {
          if (!parent.children) {
            return false;
          }

          for (var i = 0; i < parent.children.length; i++) {
            if (parent.children[i].children) {
              if (this.isUrlInChildren(parent.children[i], url)) {
                return true;
              }
            }

            if (parent.children[i].url === url || url.includes(parent.children[i].url)) {
              return true;
            }
          }

          return false;
        }
      }]);

      return NavVerticalCollapseComponent;
    }();

    NavVerticalCollapseComponent.ctorParameters = function () {
      return [{
        type: _navigation_service__WEBPACK_IMPORTED_MODULE_4__["NavigationService"]
      }, {
        type: _angular_router__WEBPACK_IMPORTED_MODULE_2__["Router"]
      }];
    };

    tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["HostBinding"])('class')], NavVerticalCollapseComponent.prototype, "classes", void 0);
    tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["HostBinding"])('class.open')], NavVerticalCollapseComponent.prototype, "isOpen", void 0);
    tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["HostBinding"])('class.selected')], NavVerticalCollapseComponent.prototype, "isSelected", void 0);
    tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Input"])()], NavVerticalCollapseComponent.prototype, "item", void 0);
    NavVerticalCollapseComponent = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Component"])({
      selector: 'app-nav-vertical-collapse',
      template: tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(
      /*! raw-loader!./nav-collapse.component.html */
      "./node_modules/raw-loader/dist/cjs.js!./src/gaxon/components/navigation/vertical/nav-collapse/nav-collapse.component.html"))["default"],
      animations: _mix_animations__WEBPACK_IMPORTED_MODULE_3__["appAnimations"],
      styles: [tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(
      /*! ./nav-collapse.component.scss */
      "./src/gaxon/components/navigation/vertical/nav-collapse/nav-collapse.component.scss"))["default"]]
    })], NavVerticalCollapseComponent);
    /***/
  },

  /***/
  "./src/gaxon/components/navigation/vertical/nav-group/nav-group.component.scss":
  /*!*************************************************************************************!*\
    !*** ./src/gaxon/components/navigation/vertical/nav-group/nav-group.component.scss ***!
    \*************************************************************************************/

  /*! exports provided: default */

  /***/
  function srcGaxonComponentsNavigationVerticalNavGroupNavGroupComponentScss(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony default export */


    __webpack_exports__["default"] = "\n/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IiIsImZpbGUiOiJzcmMvZ2F4b24vY29tcG9uZW50cy9uYXZpZ2F0aW9uL3ZlcnRpY2FsL25hdi1ncm91cC9uYXYtZ3JvdXAuY29tcG9uZW50LnNjc3MifQ== */";
    /***/
  },

  /***/
  "./src/gaxon/components/navigation/vertical/nav-group/nav-group.component.ts":
  /*!***********************************************************************************!*\
    !*** ./src/gaxon/components/navigation/vertical/nav-group/nav-group.component.ts ***!
    \***********************************************************************************/

  /*! exports provided: NavVerticalGroupComponent */

  /***/
  function srcGaxonComponentsNavigationVerticalNavGroupNavGroupComponentTs(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony export (binding) */


    __webpack_require__.d(__webpack_exports__, "NavVerticalGroupComponent", function () {
      return NavVerticalGroupComponent;
    });
    /* harmony import */


    var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(
    /*! tslib */
    "./node_modules/tslib/tslib.es6.js");
    /* harmony import */


    var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(
    /*! @angular/core */
    "./node_modules/@angular/core/fesm2015/core.js");

    var NavVerticalGroupComponent = /*#__PURE__*/function () {
      function NavVerticalGroupComponent() {
        _classCallCheck(this, NavVerticalGroupComponent);

        this.classes = 'nav-group';
      }

      _createClass(NavVerticalGroupComponent, [{
        key: "ngOnInit",
        value: function ngOnInit() {}
      }]);

      return NavVerticalGroupComponent;
    }();

    tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["HostBinding"])('class')], NavVerticalGroupComponent.prototype, "classes", void 0);
    tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Input"])()], NavVerticalGroupComponent.prototype, "item", void 0);
    NavVerticalGroupComponent = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Component"])({
      selector: 'app-nav-vertical-group',
      template: tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(
      /*! raw-loader!./nav-group.component.html */
      "./node_modules/raw-loader/dist/cjs.js!./src/gaxon/components/navigation/vertical/nav-group/nav-group.component.html"))["default"],
      styles: [tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(
      /*! ./nav-group.component.scss */
      "./src/gaxon/components/navigation/vertical/nav-group/nav-group.component.scss"))["default"]]
    })], NavVerticalGroupComponent);
    /***/
  },

  /***/
  "./src/gaxon/components/navigation/vertical/nav-item/nav-item.component.scss":
  /*!***********************************************************************************!*\
    !*** ./src/gaxon/components/navigation/vertical/nav-item/nav-item.component.scss ***!
    \***********************************************************************************/

  /*! exports provided: default */

  /***/
  function srcGaxonComponentsNavigationVerticalNavItemNavItemComponentScss(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony default export */


    __webpack_exports__["default"] = "\n/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IiIsImZpbGUiOiJzcmMvZ2F4b24vY29tcG9uZW50cy9uYXZpZ2F0aW9uL3ZlcnRpY2FsL25hdi1pdGVtL25hdi1pdGVtLmNvbXBvbmVudC5zY3NzIn0= */";
    /***/
  },

  /***/
  "./src/gaxon/components/navigation/vertical/nav-item/nav-item.component.ts":
  /*!*********************************************************************************!*\
    !*** ./src/gaxon/components/navigation/vertical/nav-item/nav-item.component.ts ***!
    \*********************************************************************************/

  /*! exports provided: NavVerticalItemComponent */

  /***/
  function srcGaxonComponentsNavigationVerticalNavItemNavItemComponentTs(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony export (binding) */


    __webpack_require__.d(__webpack_exports__, "NavVerticalItemComponent", function () {
      return NavVerticalItemComponent;
    });
    /* harmony import */


    var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(
    /*! tslib */
    "./node_modules/tslib/tslib.es6.js");
    /* harmony import */


    var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(
    /*! @angular/core */
    "./node_modules/@angular/core/fesm2015/core.js");

    var NavVerticalItemComponent = /*#__PURE__*/function () {
      function NavVerticalItemComponent() {
        _classCallCheck(this, NavVerticalItemComponent);

        this.classes = 'dt-side-nav__item';
      }

      _createClass(NavVerticalItemComponent, [{
        key: "ngOnInit",
        value: function ngOnInit() {}
      }]);

      return NavVerticalItemComponent;
    }();

    tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["HostBinding"])('class')], NavVerticalItemComponent.prototype, "classes", void 0);
    tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Input"])()], NavVerticalItemComponent.prototype, "item", void 0);
    NavVerticalItemComponent = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Component"])({
      selector: 'app-nav-vertical-item',
      template: tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(
      /*! raw-loader!./nav-item.component.html */
      "./node_modules/raw-loader/dist/cjs.js!./src/gaxon/components/navigation/vertical/nav-item/nav-item.component.html"))["default"],
      styles: [tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(
      /*! ./nav-item.component.scss */
      "./src/gaxon/components/navigation/vertical/nav-item/nav-item.component.scss"))["default"]]
    })], NavVerticalItemComponent);
    /***/
  }
}]);
//# sourceMappingURL=default~layouts-back-office-layout-back-office-layout-module~layouts-back-office-mini-layout-back-of~325f412c-es5.js.map