function _toConsumableArray(arr) { return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread(); }

function _nonIterableSpread() { throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }

function _unsupportedIterableToArray(o, minLen) { if (!o) return; if (typeof o === "string") return _arrayLikeToArray(o, minLen); var n = Object.prototype.toString.call(o).slice(8, -1); if (n === "Object" && o.constructor) n = o.constructor.name; if (n === "Map" || n === "Set") return Array.from(o); if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen); }

function _iterableToArray(iter) { if (typeof Symbol !== "undefined" && Symbol.iterator in Object(iter)) return Array.from(iter); }

function _arrayWithoutHoles(arr) { if (Array.isArray(arr)) return _arrayLikeToArray(arr); }

function _arrayLikeToArray(arr, len) { if (len == null || len > arr.length) len = arr.length; for (var i = 0, arr2 = new Array(len); i < len; i++) { arr2[i] = arr[i]; } return arr2; }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }

function _createClass(Constructor, protoProps, staticProps) { if (protoProps) _defineProperties(Constructor.prototype, protoProps); if (staticProps) _defineProperties(Constructor, staticProps); return Constructor; }

(window["webpackJsonp"] = window["webpackJsonp"] || []).push([["tables-tables-module"], {
  /***/
  "./node_modules/raw-loader/dist/cjs.js!./src/app/content/components/tables/basic-tables/basic-tables.component.html":
  /*!**************************************************************************************************************************!*\
    !*** ./node_modules/raw-loader/dist/cjs.js!./src/app/content/components/tables/basic-tables/basic-tables.component.html ***!
    \**************************************************************************************************************************/

  /*! exports provided: default */

  /***/
  function node_modulesRawLoaderDistCjsJsSrcAppContentComponentsTablesBasicTablesBasicTablesComponentHtml(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony default export */


    __webpack_exports__["default"] = "<app-page-header pageTitle=\"Basic Tables\"></app-page-header>\r\n\r\n<!-- Grid -->\r\n<div class=\"row\">\r\n\r\n  <!-- Grid Item -->\r\n  <div class=\"col-xl-6\">\r\n\r\n    <app-entry-header heading=\"Simple Table\"></app-entry-header>\r\n\r\n    <gx-card class=\"overflow-hidden\" body-class=\"p-0\">\r\n      <!-- Tables -->\r\n      <div class=\"table-responsive\">\r\n        <table class=\"table table-hover mb-0\">\r\n          <thead>\r\n          <tr>\r\n            <th scope=\"col\">#</th>\r\n            <th class=\"text-uppercase\" scope=\"col\">First Name</th>\r\n            <th class=\"text-uppercase\" scope=\"col\">Last Name</th>\r\n            <th class=\"text-uppercase\" scope=\"col\">Handle</th>\r\n          </tr>\r\n          </thead>\r\n          <tbody>\r\n          <tr *ngFor=\"let user of users; index as i\">\r\n            <th scope=\"row\">{{ i + 1 }}</th>\r\n            <td>{{ user.firstName }}</td>\r\n            <td>{{ user.lastName }}</td>\r\n            <td><a href=\"javascript:void(0)\" class=\"btn-link\">{{ user.handle }}</a></td>\r\n          </tr>\r\n          </tbody>\r\n        </table>\r\n      </div>\r\n      <!-- /tables -->\r\n\r\n    </gx-card>\r\n  </div>\r\n  <!-- /grid item -->\r\n\r\n  <!-- Grid Item -->\r\n  <div class=\"col-xl-6\">\r\n\r\n    <app-entry-header heading=\"Dark Table\"></app-entry-header>\r\n\r\n    <gx-card class=\"overflow-hidden\" body-class=\"p-0\">\r\n      <!-- Tables -->\r\n      <div class=\"table-responsive\">\r\n        <table class=\"table table-hover table-dark mb-0\">\r\n          <thead>\r\n          <tr>\r\n            <th scope=\"col\">#</th>\r\n            <th class=\"text-uppercase\" scope=\"col\">First Name</th>\r\n            <th class=\"text-uppercase\" scope=\"col\">Last Name</th>\r\n            <th class=\"text-uppercase\" scope=\"col\">Handle</th>\r\n          </tr>\r\n          </thead>\r\n          <tbody>\r\n          <tr *ngFor=\"let user of users; index as i\">\r\n            <th scope=\"row\">{{ i + 1 }}</th>\r\n            <td>{{ user.firstName }}</td>\r\n            <td>{{ user.lastName }}</td>\r\n            <td><a href=\"javascript:void(0)\" class=\"btn-link\">{{ user.handle }}</a></td>\r\n          </tr>\r\n          </tbody>\r\n        </table>\r\n      </div>\r\n      <!-- /tables -->\r\n\r\n    </gx-card>\r\n  </div>\r\n  <!-- /grid item -->\r\n\r\n  <!-- Grid Item -->\r\n  <div class=\"col-xl-6\">\r\n\r\n    <app-entry-header heading=\"Simple Table Inside Card\"></app-entry-header>\r\n\r\n    <bs-card class=\"overflow-hidden\" body-class=\"pt-0\">\r\n      <bs-card-header class=\"bg-transparent\">\r\n        <h3 class=\"card-title\">Card Title</h3>\r\n        <h5 class=\"card-subtitle mb-0\">Sub heading of card</h5>\r\n      </bs-card-header>\r\n\r\n      <!-- Tables -->\r\n      <div class=\"table-responsive\">\r\n        <table class=\"table mb-0\">\r\n          <thead>\r\n          <tr>\r\n            <th scope=\"col\">#</th>\r\n            <th class=\"text-uppercase\" scope=\"col\">First Name</th>\r\n            <th class=\"text-uppercase\" scope=\"col\">Last Name</th>\r\n            <th class=\"text-uppercase\" scope=\"col\">Handle</th>\r\n          </tr>\r\n          </thead>\r\n          <tbody>\r\n          <tr *ngFor=\"let user of users; index as i\">\r\n            <th scope=\"row\">{{ i + 1 }}</th>\r\n            <td>{{ user.firstName }}</td>\r\n            <td>{{ user.lastName }}</td>\r\n            <td><a href=\"javascript:void(0)\" class=\"btn-link\">{{ user.handle }}</a></td>\r\n          </tr>\r\n          </tbody>\r\n        </table>\r\n      </div>\r\n      <!-- /tables -->\r\n\r\n    </bs-card>\r\n  </div>\r\n  <!-- /grid item -->\r\n\r\n  <!-- Grid Item -->\r\n  <div class=\"col-xl-6\">\r\n\r\n    <app-entry-header heading=\"Simple Table Inside Dark Card\"></app-entry-header>\r\n\r\n    <bs-card class=\"overflow-hidden table-dark\" body-class=\"pt-0\">\r\n      <bs-card-header class=\"bg-transparent\">\r\n        <h3 class=\"card-title\">Card Title</h3>\r\n        <h5 class=\"card-subtitle mb-0\">Sub heading of card</h5>\r\n      </bs-card-header>\r\n\r\n      <!-- Tables -->\r\n      <div class=\"table-responsive\">\r\n        <table class=\"table mb-0\">\r\n          <thead>\r\n          <tr>\r\n            <th scope=\"col\">#</th>\r\n            <th class=\"text-uppercase\" scope=\"col\">First Name</th>\r\n            <th class=\"text-uppercase\" scope=\"col\">Last Name</th>\r\n            <th class=\"text-uppercase\" scope=\"col\">Handle</th>\r\n          </tr>\r\n          </thead>\r\n          <tbody>\r\n          <tr *ngFor=\"let user of users; index as i\">\r\n            <th scope=\"row\">{{ i + 1 }}</th>\r\n            <td>{{ user.firstName }}</td>\r\n            <td>{{ user.lastName }}</td>\r\n            <td><a href=\"javascript:void(0)\" class=\"btn-link\">{{ user.handle }}</a></td>\r\n          </tr>\r\n          </tbody>\r\n        </table>\r\n      </div>\r\n      <!-- /tables -->\r\n\r\n    </bs-card>\r\n  </div>\r\n  <!-- /grid item -->\r\n\r\n  <!-- Grid Item -->\r\n  <div class=\"col-xl-6\">\r\n\r\n    <app-entry-header heading=\"Simple Table With Stripped Rows\"></app-entry-header>\r\n\r\n    <gx-card class=\"overflow-hidden\" body-class=\"p-0\">\r\n      <!-- Tables -->\r\n      <div class=\"table-responsive\">\r\n        <table class=\"table table-striped mb-0\">\r\n          <thead class=\"thead-light\">\r\n          <tr>\r\n            <th scope=\"col\">#</th>\r\n            <th class=\"text-uppercase\" scope=\"col\">First Name</th>\r\n            <th class=\"text-uppercase\" scope=\"col\">Last Name</th>\r\n            <th class=\"text-uppercase\" scope=\"col\">Handle</th>\r\n          </tr>\r\n          </thead>\r\n          <tbody>\r\n          <tr *ngFor=\"let user of users; index as i\">\r\n            <th scope=\"row\">{{ i + 1 }}</th>\r\n            <td>{{ user.firstName }}</td>\r\n            <td>{{ user.lastName }}</td>\r\n            <td><a href=\"javascript:void(0)\" class=\"btn-link\">{{ user.handle }}</a></td>\r\n          </tr>\r\n          </tbody>\r\n        </table>\r\n      </div>\r\n      <!-- /tables -->\r\n\r\n    </gx-card>\r\n  </div>\r\n  <!-- /grid item -->\r\n\r\n  <!-- Grid Item -->\r\n  <div class=\"col-xl-6\">\r\n\r\n    <app-entry-header heading=\"Dark Table With Stripped Rows\"></app-entry-header>\r\n\r\n    <gx-card class=\"overflow-hidden\" body-class=\"p-0\">\r\n      <!-- Tables -->\r\n      <div class=\"table-responsive\">\r\n        <table class=\"table table-striped table-dark mb-0\">\r\n          <thead class=\"thead-dark\">\r\n          <tr>\r\n            <th scope=\"col\">#</th>\r\n            <th class=\"text-uppercase\" scope=\"col\">First Name</th>\r\n            <th class=\"text-uppercase\" scope=\"col\">Last Name</th>\r\n            <th class=\"text-uppercase\" scope=\"col\">Handle</th>\r\n          </tr>\r\n          </thead>\r\n          <tbody>\r\n          <tr *ngFor=\"let user of users; index as i\">\r\n            <th scope=\"row\">{{ i + 1 }}</th>\r\n            <td>{{ user.firstName }}</td>\r\n            <td>{{ user.lastName }}</td>\r\n            <td><a href=\"javascript:void(0)\" class=\"btn-link\">{{ user.handle }}</a></td>\r\n          </tr>\r\n          </tbody>\r\n        </table>\r\n      </div>\r\n      <!-- /tables -->\r\n\r\n    </gx-card>\r\n  </div>\r\n  <!-- /grid item -->\r\n\r\n  <!-- Grid Item -->\r\n  <div class=\"col-xl-6\">\r\n\r\n    <app-entry-header heading=\"Simple Table With Borders\"></app-entry-header>\r\n\r\n    <bs-card class=\"overflow-hidden\" body-class=\"pt-0\">\r\n      <!-- Card Header -->\r\n      <bs-card-header class=\"bg-transparent\">\r\n        <h3 class=\"card-title\">Card Title</h3>\r\n        <h5 class=\"card-subtitle mb-0\">Sub heading of card</h5>\r\n      </bs-card-header>\r\n      <!-- /card header -->\r\n\r\n      <!-- Tables -->\r\n      <div class=\"table-responsive\">\r\n        <table class=\"table table-bordered mb-0\">\r\n          <thead>\r\n          <tr>\r\n            <th scope=\"col\">#</th>\r\n            <th class=\"text-uppercase\" scope=\"col\">First Name</th>\r\n            <th class=\"text-uppercase\" scope=\"col\">Last Name</th>\r\n            <th class=\"text-uppercase\" scope=\"col\">Handle</th>\r\n          </tr>\r\n          </thead>\r\n          <tbody>\r\n          <tr *ngFor=\"let user of users; index as i\">\r\n            <th scope=\"row\">{{ i + 1 }}</th>\r\n            <td>{{ user.firstName }}</td>\r\n            <td>{{ user.lastName }}</td>\r\n            <td><a href=\"javascript:void(0)\" class=\"btn-link\">{{ user.handle }}</a></td>\r\n          </tr>\r\n          </tbody>\r\n        </table>\r\n      </div>\r\n      <!-- /tables -->\r\n\r\n    </bs-card>\r\n  </div>\r\n  <!-- /grid item -->\r\n\r\n  <!-- Grid Item -->\r\n  <div class=\"col-xl-6\">\r\n\r\n    <app-entry-header heading=\"Simple Table With Borders\"></app-entry-header>\r\n\r\n    <bs-card class=\"overflow-hidden table-dark\" body-class=\"pt-0\">\r\n      <!-- Card Header -->\r\n      <bs-card-header class=\"bg-transparent\">\r\n        <h3 class=\"card-title\">Card Title</h3>\r\n        <h5 class=\"card-subtitle mb-0\">Sub heading of card</h5>\r\n      </bs-card-header>\r\n      <!-- /card header -->\r\n\r\n      <!-- Tables -->\r\n      <div class=\"table-responsive\">\r\n        <table class=\"table table-bordered table-dark mb-0\">\r\n          <thead>\r\n          <tr>\r\n            <th scope=\"col\">#</th>\r\n            <th class=\"text-uppercase\" scope=\"col\">First Name</th>\r\n            <th class=\"text-uppercase\" scope=\"col\">Last Name</th>\r\n            <th class=\"text-uppercase\" scope=\"col\">Handle</th>\r\n          </tr>\r\n          </thead>\r\n          <tbody>\r\n          <tr *ngFor=\"let user of users; index as i\">\r\n            <th scope=\"row\">{{ i + 1 }}</th>\r\n            <td>{{ user.firstName }}</td>\r\n            <td>{{ user.lastName }}</td>\r\n            <td><a href=\"javascript:void(0)\" class=\"btn-link\">{{ user.handle }}</a></td>\r\n          </tr>\r\n          </tbody>\r\n        </table>\r\n      </div>\r\n      <!-- /tables -->\r\n\r\n    </bs-card>\r\n  </div>\r\n  <!-- /grid item -->\r\n\r\n</div>\r\n<!-- /grid -->\r\n";
    /***/
  },

  /***/
  "./node_modules/raw-loader/dist/cjs.js!./src/app/content/components/tables/data-tables/data-tables.component.html":
  /*!************************************************************************************************************************!*\
    !*** ./node_modules/raw-loader/dist/cjs.js!./src/app/content/components/tables/data-tables/data-tables.component.html ***!
    \************************************************************************************************************************/

  /*! exports provided: default */

  /***/
  function node_modulesRawLoaderDistCjsJsSrcAppContentComponentsTablesDataTablesDataTablesComponentHtml(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony default export */


    __webpack_exports__["default"] = "<app-page-header pageTitle=\"Data Tables\"></app-page-header>\r\n\r\n<!-- Grid -->\r\n<div class=\"row\">\r\n\r\n  <!-- Grid Item -->\r\n  <div class=\"col-xl-12\">\r\n    <app-entry-header heading=\"Sortable table\"></app-entry-header>\r\n\r\n    <gx-card>\r\n      <!-- Tables -->\r\n      <div class=\"table-responsive\">\r\n        <app-table-sortable></app-table-sortable>\r\n      </div>\r\n      <!-- /tables -->\r\n\r\n    </gx-card>\r\n\r\n    <app-entry-header heading=\"Search and filtering\"></app-entry-header>\r\n\r\n    <gx-card>\r\n      <!-- Tables -->\r\n      <div class=\"table-responsive\">\r\n        <app-table-filtering></app-table-filtering>\r\n      </div>\r\n      <!-- /tables -->\r\n\r\n    </gx-card>\r\n\r\n    <app-entry-header heading=\"Pagination\"></app-entry-header>\r\n\r\n    <gx-card>\r\n      <!-- Tables -->\r\n      <div class=\"table-responsive\">\r\n        <app-table-pagination></app-table-pagination>\r\n      </div>\r\n      <!-- /tables -->\r\n\r\n    </gx-card>\r\n\r\n    <app-entry-header heading=\"Complete example\"></app-entry-header>\r\n\r\n    <gx-card>\r\n      <!-- Tables -->\r\n      <div class=\"table-responsive\">\r\n        <app-table-complete></app-table-complete>\r\n      </div>\r\n      <!-- /tables -->\r\n\r\n    </gx-card>\r\n\r\n  </div>\r\n  <!-- /grid item -->\r\n\r\n</div>\r\n<!-- /grid -->\r\n";
    /***/
  },

  /***/
  "./node_modules/raw-loader/dist/cjs.js!./src/app/content/components/tables/data-tables/table-complete/table-complete.component.html":
  /*!******************************************************************************************************************************************!*\
    !*** ./node_modules/raw-loader/dist/cjs.js!./src/app/content/components/tables/data-tables/table-complete/table-complete.component.html ***!
    \******************************************************************************************************************************************/

  /*! exports provided: default */

  /***/
  function node_modulesRawLoaderDistCjsJsSrcAppContentComponentsTablesDataTablesTableCompleteTableCompleteComponentHtml(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony default export */


    __webpack_exports__["default"] = "<form>\r\n  <div class=\"dataTables_wrapper dt-bootstrap4\">\r\n    <div class=\"dataTables_length\">\r\n      <label>\r\n        Show\r\n        <select class=\"custom-select\" style=\"width: auto\" name=\"pageSize\" [(ngModel)]=\"service.pageSize\">\r\n          <option [ngValue]=\"2\">2</option>\r\n          <option [ngValue]=\"4\">4</option>\r\n          <option [ngValue]=\"6\">6</option>\r\n        </select>\r\n        Entries\r\n      </label>\r\n    </div>\r\n    <div class=\"dataTables_filter form-group form-inline\">\r\n      <label>Search:\r\n        <input class=\"form-control form-control-sm ml-1\" type=\"text\" name=\"searchTerm\" [(ngModel)]=\"service.searchTerm\"/>\r\n      </label>\r\n    </div>\r\n\r\n    <div class=\"dataTables_info mb-2\">Showing {{service.pageSize * (service.page - 1)}} to {{service.pageSize * (service.page - 1) + (countries$ | async).length}} of {{total$ | async}} entries</div>\r\n  </div>\r\n\r\n  <div class=\"my-5\" *ngIf=\"service.loading$ | async\"><span>Loading...</span></div>\r\n\r\n  <table class=\"table table-striped table-bordered table-hover\">\r\n    <thead>\r\n    <tr>\r\n      <th scope=\"col\">#</th>\r\n      <th scope=\"col\" sortable=\"name\" (sort)=\"onSort($event)\">Country</th>\r\n      <th scope=\"col\" sortable=\"area\" (sort)=\"onSort($event)\">Area</th>\r\n      <th scope=\"col\" sortable=\"population\" (sort)=\"onSort($event)\">Population</th>\r\n    </tr>\r\n    </thead>\r\n    <tbody>\r\n    <tr *ngFor=\"let country of countries$ | async\">\r\n      <th scope=\"row\">{{ country.id }}</th>\r\n      <td>\r\n        <img [src]=\"'https://upload.wikimedia.org/wikipedia/commons/' + country.flag\" class=\"mr-2\" style=\"width: 20px\">\r\n        <ngb-highlight [result]=\"country.name\" [term]=\"service.searchTerm\"></ngb-highlight>\r\n      </td>\r\n      <td>\r\n        <ngb-highlight [result]=\"country.area | number\" [term]=\"service.searchTerm\"></ngb-highlight>\r\n      </td>\r\n      <td>\r\n        <ngb-highlight [result]=\"country.population | number\" [term]=\"service.searchTerm\"></ngb-highlight>\r\n      </td>\r\n    </tr>\r\n    </tbody>\r\n  </table>\r\n\r\n  <div class=\"d-flex justify-content-center p-2\">\r\n    <ngb-pagination\r\n      [collectionSize]=\"total$ | async\" [(page)]=\"service.page\" [pageSize]=\"service.pageSize\">\r\n    </ngb-pagination>\r\n  </div>\r\n\r\n</form>\r\n";
    /***/
  },

  /***/
  "./node_modules/raw-loader/dist/cjs.js!./src/app/content/components/tables/data-tables/table-filtering/table-filtering.component.html":
  /*!********************************************************************************************************************************************!*\
    !*** ./node_modules/raw-loader/dist/cjs.js!./src/app/content/components/tables/data-tables/table-filtering/table-filtering.component.html ***!
    \********************************************************************************************************************************************/

  /*! exports provided: default */

  /***/
  function node_modulesRawLoaderDistCjsJsSrcAppContentComponentsTablesDataTablesTableFilteringTableFilteringComponentHtml(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony default export */


    __webpack_exports__["default"] = "<form>\r\n  <div class=\"form-group form-inline\">\r\n    Full text search: <input class=\"form-control ml-2\" type=\"text\" [formControl]=\"filter\"/>\r\n  </div>\r\n</form>\r\n\r\n<table class=\"table table-striped table-bordered table-hover\">\r\n  <thead>\r\n  <tr>\r\n    <th scope=\"col\">#</th>\r\n    <th scope=\"col\">Country</th>\r\n    <th scope=\"col\">Area</th>\r\n    <th scope=\"col\">Population</th>\r\n  </tr>\r\n  </thead>\r\n  <tbody>\r\n  <tr *ngFor=\"let country of countries$ | async; index as i\">\r\n    <th scope=\"row\">{{ i + 1 }}</th>\r\n    <td>\r\n      <img [src]=\"'https://upload.wikimedia.org/wikipedia/commons/' + country.flag\" class=\"mr-2\" style=\"width: 20px\">\r\n      <ngb-highlight [result]=\"country.name\" [term]=\"filter.value\"></ngb-highlight>\r\n    </td>\r\n    <td>\r\n      <ngb-highlight [result]=\"country.area | number\" [term]=\"filter.value\"></ngb-highlight>\r\n    </td>\r\n    <td>\r\n      <ngb-highlight [result]=\"country.population | number\" [term]=\"filter.value\"></ngb-highlight>\r\n    </td>\r\n  </tr>\r\n  </tbody>\r\n</table>\r\n";
    /***/
  },

  /***/
  "./node_modules/raw-loader/dist/cjs.js!./src/app/content/components/tables/data-tables/table-pagination/table-pagination.component.html":
  /*!**********************************************************************************************************************************************!*\
    !*** ./node_modules/raw-loader/dist/cjs.js!./src/app/content/components/tables/data-tables/table-pagination/table-pagination.component.html ***!
    \**********************************************************************************************************************************************/

  /*! exports provided: default */

  /***/
  function node_modulesRawLoaderDistCjsJsSrcAppContentComponentsTablesDataTablesTablePaginationTablePaginationComponentHtml(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony default export */


    __webpack_exports__["default"] = "<table class=\"table table-striped table-bordered table-hover\">\r\n  <thead>\r\n  <tr>\r\n    <th scope=\"col\">#</th>\r\n    <th scope=\"col\">Country</th>\r\n    <th scope=\"col\">Area</th>\r\n    <th scope=\"col\">Population</th>\r\n  </tr>\r\n  </thead>\r\n  <tbody>\r\n  <tr *ngFor=\"let country of countries\">\r\n    <th scope=\"row\">{{ country.id }}</th>\r\n    <td>\r\n      <img [src]=\"'https://upload.wikimedia.org/wikipedia/commons/' + country.flag\" class=\"mr-2\" style=\"width: 20px\">\r\n      {{ country.name }}\r\n    </td>\r\n    <td>{{ country.area | number}}</td>\r\n    <td>{{ country.population | number }}</td>\r\n  </tr>\r\n  </tbody>\r\n</table>\r\n\r\n<div class=\"d-flex justify-content-between p-2\">\r\n  <ngb-pagination [collectionSize]=\"collectionSize\" [(page)]=\"page\" [pageSize]=\"pageSize\">\r\n  </ngb-pagination>\r\n\r\n  <select class=\"custom-select\" style=\"width: auto\" [(ngModel)]=\"pageSize\">\r\n    <option [ngValue]=\"2\">2 items per page</option>\r\n    <option [ngValue]=\"4\">4 items per page</option>\r\n    <option [ngValue]=\"6\">6 items per page</option>\r\n  </select>\r\n</div>\r\n";
    /***/
  },

  /***/
  "./node_modules/raw-loader/dist/cjs.js!./src/app/content/components/tables/data-tables/table-sortable/table-sortable.component.html":
  /*!******************************************************************************************************************************************!*\
    !*** ./node_modules/raw-loader/dist/cjs.js!./src/app/content/components/tables/data-tables/table-sortable/table-sortable.component.html ***!
    \******************************************************************************************************************************************/

  /*! exports provided: default */

  /***/
  function node_modulesRawLoaderDistCjsJsSrcAppContentComponentsTablesDataTablesTableSortableTableSortableComponentHtml(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony default export */


    __webpack_exports__["default"] = "<table class=\"table table-striped table-bordered table-hover\">\r\n  <thead>\r\n  <tr>\r\n    <th scope=\"col\">#</th>\r\n    <th scope=\"col\" sortable=\"name\" (sort)=\"onSort($event)\">Country</th>\r\n    <th scope=\"col\" sortable=\"area\" (sort)=\"onSort($event)\">Area</th>\r\n    <th scope=\"col\" sortable=\"population\" (sort)=\"onSort($event)\">Population</th>\r\n  </tr>\r\n  </thead>\r\n  <tbody>\r\n  <tr *ngFor=\"let country of countries\" class=\"gradeX\">\r\n    <th scope=\"row\">{{ country.id }}</th>\r\n    <td>\r\n      <img [src]=\"'https://upload.wikimedia.org/wikipedia/commons/' + country.flag\" class=\"mr-2\" style=\"width: 20px\">\r\n      {{ country.name }}\r\n    </td>\r\n    <td>{{ country.area | number }}</td>\r\n    <td>{{ country.population | number }}</td>\r\n  </tr>\r\n  </tbody>\r\n</table>\r\n";
    /***/
  },

  /***/
  "./src/app/content/components/tables/basic-tables/basic-tables.component.scss":
  /*!************************************************************************************!*\
    !*** ./src/app/content/components/tables/basic-tables/basic-tables.component.scss ***!
    \************************************************************************************/

  /*! exports provided: default */

  /***/
  function srcAppContentComponentsTablesBasicTablesBasicTablesComponentScss(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony default export */


    __webpack_exports__["default"] = "\n/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IiIsImZpbGUiOiJzcmMvYXBwL2NvbnRlbnQvY29tcG9uZW50cy90YWJsZXMvYmFzaWMtdGFibGVzL2Jhc2ljLXRhYmxlcy5jb21wb25lbnQuc2NzcyJ9 */";
    /***/
  },

  /***/
  "./src/app/content/components/tables/basic-tables/basic-tables.component.ts":
  /*!**********************************************************************************!*\
    !*** ./src/app/content/components/tables/basic-tables/basic-tables.component.ts ***!
    \**********************************************************************************/

  /*! exports provided: BasicTablesComponent */

  /***/
  function srcAppContentComponentsTablesBasicTablesBasicTablesComponentTs(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony export (binding) */


    __webpack_require__.d(__webpack_exports__, "BasicTablesComponent", function () {
      return BasicTablesComponent;
    });
    /* harmony import */


    var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(
    /*! tslib */
    "./node_modules/tslib/tslib.es6.js");
    /* harmony import */


    var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(
    /*! @angular/core */
    "./node_modules/@angular/core/fesm2015/core.js");

    var USERS = [{
      firstName: 'Chris',
      lastName: 'Crowder',
      handle: '@c.crowder'
    }, {
      firstName: 'Domnic',
      lastName: 'Brown',
      handle: '@dom.brown'
    }, {
      firstName: 'Jonathan',
      lastName: 'Madano',
      handle: '@john.madano'
    }, {
      firstName: 'Stella',
      lastName: 'Johnson',
      handle: '@stella.johnson'
    }, {
      firstName: 'Michael',
      lastName: 'Jacklin',
      handle: '@michael.jack'
    }];

    var BasicTablesComponent = /*#__PURE__*/function () {
      function BasicTablesComponent() {
        _classCallCheck(this, BasicTablesComponent);

        this.users = USERS;
      }

      _createClass(BasicTablesComponent, [{
        key: "ngOnInit",
        value: function ngOnInit() {}
      }]);

      return BasicTablesComponent;
    }();

    BasicTablesComponent = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Component"])({
      selector: 'app-basic',
      template: tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(
      /*! raw-loader!./basic-tables.component.html */
      "./node_modules/raw-loader/dist/cjs.js!./src/app/content/components/tables/basic-tables/basic-tables.component.html"))["default"],
      styles: [tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(
      /*! ./basic-tables.component.scss */
      "./src/app/content/components/tables/basic-tables/basic-tables.component.scss"))["default"]]
    })], BasicTablesComponent);
    /***/
  },

  /***/
  "./src/app/content/components/tables/data-tables/countries.ts":
  /*!********************************************************************!*\
    !*** ./src/app/content/components/tables/data-tables/countries.ts ***!
    \********************************************************************/

  /*! exports provided: COUNTRIES */

  /***/
  function srcAppContentComponentsTablesDataTablesCountriesTs(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony export (binding) */


    __webpack_require__.d(__webpack_exports__, "COUNTRIES", function () {
      return COUNTRIES;
    });
    /* harmony import */


    var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(
    /*! tslib */
    "./node_modules/tslib/tslib.es6.js");

    var COUNTRIES = [{
      id: 1,
      name: 'Russia',
      flag: 'f/f3/Flag_of_Russia.svg',
      area: 17075200,
      population: 146989754
    }, {
      id: 2,
      name: 'France',
      flag: 'c/c3/Flag_of_France.svg',
      area: 640679,
      population: 64979548
    }, {
      id: 3,
      name: 'Germany',
      flag: 'b/ba/Flag_of_Germany.svg',
      area: 357114,
      population: 82114224
    }, {
      id: 4,
      name: 'Portugal',
      flag: '5/5c/Flag_of_Portugal.svg',
      area: 92090,
      population: 10329506
    }, {
      id: 5,
      name: 'Canada',
      flag: 'c/cf/Flag_of_Canada.svg',
      area: 9976140,
      population: 36624199
    }, {
      id: 6,
      name: 'Vietnam',
      flag: '2/21/Flag_of_Vietnam.svg',
      area: 331212,
      population: 95540800
    }, {
      id: 7,
      name: 'Brazil',
      flag: '0/05/Flag_of_Brazil.svg',
      area: 8515767,
      population: 209288278
    }, {
      id: 8,
      name: 'Mexico',
      flag: 'f/fc/Flag_of_Mexico.svg',
      area: 1964375,
      population: 129163276
    }, {
      id: 9,
      name: 'United States',
      flag: 'a/a4/Flag_of_the_United_States.svg',
      area: 9629091,
      population: 324459463
    }, {
      id: 10,
      name: 'India',
      flag: '4/41/Flag_of_India.svg',
      area: 3287263,
      population: 1324171354
    }, {
      id: 11,
      name: 'Indonesia',
      flag: '9/9f/Flag_of_Indonesia.svg',
      area: 1910931,
      population: 263991379
    }, {
      id: 12,
      name: 'Tuvalu',
      flag: '3/38/Flag_of_Tuvalu.svg',
      area: 26,
      population: 11097
    }, {
      id: 13,
      name: 'China',
      flag: 'f/fa/Flag_of_the_People%27s_Republic_of_China.svg',
      area: 9596960,
      population: 1409517397
    }];
    /***/
  },

  /***/
  "./src/app/content/components/tables/data-tables/country.service.ts":
  /*!**************************************************************************!*\
    !*** ./src/app/content/components/tables/data-tables/country.service.ts ***!
    \**************************************************************************/

  /*! exports provided: CountryService */

  /***/
  function srcAppContentComponentsTablesDataTablesCountryServiceTs(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony export (binding) */


    __webpack_require__.d(__webpack_exports__, "CountryService", function () {
      return CountryService;
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


    var rxjs__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(
    /*! rxjs */
    "./node_modules/rxjs/_esm2015/index.js");
    /* harmony import */


    var rxjs_operators__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(
    /*! rxjs/operators */
    "./node_modules/rxjs/_esm2015/operators/index.js");
    /* harmony import */


    var _countries__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(
    /*! ./countries */
    "./src/app/content/components/tables/data-tables/countries.ts");

    function compare(v1, v2) {
      return v1 < v2 ? -1 : v1 > v2 ? 1 : 0;
    }

    function sort(countries, column, direction) {
      if (direction === '') {
        return countries;
      } else {
        return _toConsumableArray(countries).sort(function (a, b) {
          var res = compare(a[column], b[column]);
          return direction === 'asc' ? res : -res;
        });
      }
    }

    function matches(country, term, pipe) {
      return country.name.toLowerCase().includes(term) || pipe.transform(country.area).includes(term) || pipe.transform(country.population).includes(term);
    }

    var CountryService = /*#__PURE__*/function () {
      function CountryService(pipe) {
        var _this = this;

        _classCallCheck(this, CountryService);

        this.pipe = pipe;
        this._loading$ = new rxjs__WEBPACK_IMPORTED_MODULE_3__["BehaviorSubject"](true);
        this._search$ = new rxjs__WEBPACK_IMPORTED_MODULE_3__["Subject"]();
        this._countries$ = new rxjs__WEBPACK_IMPORTED_MODULE_3__["BehaviorSubject"]([]);
        this._total$ = new rxjs__WEBPACK_IMPORTED_MODULE_3__["BehaviorSubject"](0);
        this._state = {
          page: 1,
          pageSize: 4,
          searchTerm: '',
          sortColumn: '',
          sortDirection: ''
        };

        this._search$.pipe(Object(rxjs_operators__WEBPACK_IMPORTED_MODULE_4__["tap"])(function () {
          return _this._loading$.next(true);
        }), Object(rxjs_operators__WEBPACK_IMPORTED_MODULE_4__["debounceTime"])(200), Object(rxjs_operators__WEBPACK_IMPORTED_MODULE_4__["switchMap"])(function () {
          return _this._search();
        }), Object(rxjs_operators__WEBPACK_IMPORTED_MODULE_4__["delay"])(200), Object(rxjs_operators__WEBPACK_IMPORTED_MODULE_4__["tap"])(function () {
          return _this._loading$.next(false);
        })).subscribe(function (result) {
          _this._countries$.next(result.countries);

          _this._total$.next(result.total);
        });

        this._search$.next();
      }

      _createClass(CountryService, [{
        key: "_set",
        value: function _set(patch) {
          Object.assign(this._state, patch);

          this._search$.next();
        }
      }, {
        key: "_search",
        value: function _search() {
          var _this2 = this;

          var _this$_state = this._state,
              sortColumn = _this$_state.sortColumn,
              sortDirection = _this$_state.sortDirection,
              pageSize = _this$_state.pageSize,
              page = _this$_state.page,
              searchTerm = _this$_state.searchTerm; // 1. sort

          var countries = sort(_countries__WEBPACK_IMPORTED_MODULE_5__["COUNTRIES"], sortColumn, sortDirection); // 2. filter

          countries = countries.filter(function (country) {
            return matches(country, searchTerm, _this2.pipe);
          });
          var total = countries.length; // 3. paginate

          countries = countries.slice((page - 1) * pageSize, (page - 1) * pageSize + pageSize);
          return Object(rxjs__WEBPACK_IMPORTED_MODULE_3__["of"])({
            countries: countries,
            total: total
          });
        }
      }, {
        key: "countries$",
        get: function get() {
          return this._countries$.asObservable();
        }
      }, {
        key: "total$",
        get: function get() {
          return this._total$.asObservable();
        }
      }, {
        key: "loading$",
        get: function get() {
          return this._loading$.asObservable();
        }
      }, {
        key: "page",
        get: function get() {
          return this._state.page;
        },
        set: function set(page) {
          this._set({
            page: page
          });
        }
      }, {
        key: "pageSize",
        get: function get() {
          return this._state.pageSize;
        },
        set: function set(pageSize) {
          this._set({
            pageSize: pageSize
          });
        }
      }, {
        key: "searchTerm",
        get: function get() {
          return this._state.searchTerm;
        },
        set: function set(searchTerm) {
          this._set({
            searchTerm: searchTerm
          });
        }
      }, {
        key: "sortColumn",
        set: function set(sortColumn) {
          this._set({
            sortColumn: sortColumn
          });
        }
      }, {
        key: "sortDirection",
        set: function set(sortDirection) {
          this._set({
            sortDirection: sortDirection
          });
        }
      }]);

      return CountryService;
    }();

    CountryService.ctorParameters = function () {
      return [{
        type: _angular_common__WEBPACK_IMPORTED_MODULE_2__["DecimalPipe"]
      }];
    };

    CountryService = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Injectable"])({
      providedIn: 'root'
    })], CountryService);
    /***/
  },

  /***/
  "./src/app/content/components/tables/data-tables/data-tables.component.scss":
  /*!**********************************************************************************!*\
    !*** ./src/app/content/components/tables/data-tables/data-tables.component.scss ***!
    \**********************************************************************************/

  /*! exports provided: default */

  /***/
  function srcAppContentComponentsTablesDataTablesDataTablesComponentScss(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony default export */


    __webpack_exports__["default"] = "\n/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IiIsImZpbGUiOiJzcmMvYXBwL2NvbnRlbnQvY29tcG9uZW50cy90YWJsZXMvZGF0YS10YWJsZXMvZGF0YS10YWJsZXMuY29tcG9uZW50LnNjc3MifQ== */";
    /***/
  },

  /***/
  "./src/app/content/components/tables/data-tables/data-tables.component.ts":
  /*!********************************************************************************!*\
    !*** ./src/app/content/components/tables/data-tables/data-tables.component.ts ***!
    \********************************************************************************/

  /*! exports provided: DataTablesComponent */

  /***/
  function srcAppContentComponentsTablesDataTablesDataTablesComponentTs(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony export (binding) */


    __webpack_require__.d(__webpack_exports__, "DataTablesComponent", function () {
      return DataTablesComponent;
    });
    /* harmony import */


    var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(
    /*! tslib */
    "./node_modules/tslib/tslib.es6.js");
    /* harmony import */


    var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(
    /*! @angular/core */
    "./node_modules/@angular/core/fesm2015/core.js");

    var DataTablesComponent = /*#__PURE__*/function () {
      function DataTablesComponent() {
        _classCallCheck(this, DataTablesComponent);
      }

      _createClass(DataTablesComponent, [{
        key: "ngOnInit",
        value: function ngOnInit() {}
      }]);

      return DataTablesComponent;
    }();

    DataTablesComponent = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Component"])({
      selector: 'app-data',
      template: tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(
      /*! raw-loader!./data-tables.component.html */
      "./node_modules/raw-loader/dist/cjs.js!./src/app/content/components/tables/data-tables/data-tables.component.html"))["default"],
      styles: [tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(
      /*! ./data-tables.component.scss */
      "./src/app/content/components/tables/data-tables/data-tables.component.scss"))["default"]]
    })], DataTablesComponent);
    /***/
  },

  /***/
  "./src/app/content/components/tables/data-tables/sortable.directive.ts":
  /*!*****************************************************************************!*\
    !*** ./src/app/content/components/tables/data-tables/sortable.directive.ts ***!
    \*****************************************************************************/

  /*! exports provided: SortableDirective */

  /***/
  function srcAppContentComponentsTablesDataTablesSortableDirectiveTs(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony export (binding) */


    __webpack_require__.d(__webpack_exports__, "SortableDirective", function () {
      return SortableDirective;
    });
    /* harmony import */


    var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(
    /*! tslib */
    "./node_modules/tslib/tslib.es6.js");
    /* harmony import */


    var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(
    /*! @angular/core */
    "./node_modules/@angular/core/fesm2015/core.js");

    var _rotate = {
      'asc': 'desc',
      'desc': '',
      '': 'asc'
    };

    var SortableDirective = /*#__PURE__*/function () {
      function SortableDirective() {
        _classCallCheck(this, SortableDirective);

        this.direction = '';
        this.sort = new _angular_core__WEBPACK_IMPORTED_MODULE_1__["EventEmitter"]();
      }

      _createClass(SortableDirective, [{
        key: "onClick",
        value: function onClick() {
          this.rotate();
        }
      }, {
        key: "rotate",
        value: function rotate() {
          this.direction = _rotate[this.direction];
          this.sort.emit({
            column: this.sortable,
            direction: this.direction
          });
        }
      }]);

      return SortableDirective;
    }();

    tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Input"])()], SortableDirective.prototype, "sortable", void 0);
    tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Input"])()], SortableDirective.prototype, "direction", void 0);
    tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Output"])()], SortableDirective.prototype, "sort", void 0);
    tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["HostListener"])('click')], SortableDirective.prototype, "onClick", null);
    SortableDirective = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Directive"])({
      selector: 'th[sortable]',
      host: {
        '[class.asc]': 'direction === "asc"',
        '[class.desc]': 'direction === "desc"'
      }
    })], SortableDirective);
    /***/
  },

  /***/
  "./src/app/content/components/tables/data-tables/table-complete/table-complete.component.scss":
  /*!****************************************************************************************************!*\
    !*** ./src/app/content/components/tables/data-tables/table-complete/table-complete.component.scss ***!
    \****************************************************************************************************/

  /*! exports provided: default */

  /***/
  function srcAppContentComponentsTablesDataTablesTableCompleteTableCompleteComponentScss(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony default export */


    __webpack_exports__["default"] = "\n/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IiIsImZpbGUiOiJzcmMvYXBwL2NvbnRlbnQvY29tcG9uZW50cy90YWJsZXMvZGF0YS10YWJsZXMvdGFibGUtY29tcGxldGUvdGFibGUtY29tcGxldGUuY29tcG9uZW50LnNjc3MifQ== */";
    /***/
  },

  /***/
  "./src/app/content/components/tables/data-tables/table-complete/table-complete.component.ts":
  /*!**************************************************************************************************!*\
    !*** ./src/app/content/components/tables/data-tables/table-complete/table-complete.component.ts ***!
    \**************************************************************************************************/

  /*! exports provided: TableCompleteComponent */

  /***/
  function srcAppContentComponentsTablesDataTablesTableCompleteTableCompleteComponentTs(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony export (binding) */


    __webpack_require__.d(__webpack_exports__, "TableCompleteComponent", function () {
      return TableCompleteComponent;
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


    var _country_service__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(
    /*! ./../country.service */
    "./src/app/content/components/tables/data-tables/country.service.ts");
    /* harmony import */


    var _sortable_directive__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(
    /*! ./../sortable.directive */
    "./src/app/content/components/tables/data-tables/sortable.directive.ts");

    var TableCompleteComponent = /*#__PURE__*/function () {
      function TableCompleteComponent(service) {
        _classCallCheck(this, TableCompleteComponent);

        this.service = service;
        this.countries$ = service.countries$;
        this.total$ = service.total$;
        console.log('total: ', service.total$);
      }

      _createClass(TableCompleteComponent, [{
        key: "onSort",
        value: function onSort(_ref) {
          var column = _ref.column,
              direction = _ref.direction;
          // resetting other headers
          this.headers.forEach(function (header) {
            if (header.sortable !== column) {
              header.direction = '';
            }
          });
          this.service.sortColumn = column;
          this.service.sortDirection = direction;
        }
      }]);

      return TableCompleteComponent;
    }();

    TableCompleteComponent.ctorParameters = function () {
      return [{
        type: _country_service__WEBPACK_IMPORTED_MODULE_3__["CountryService"]
      }];
    };

    tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["ViewChildren"])(_sortable_directive__WEBPACK_IMPORTED_MODULE_4__["SortableDirective"])], TableCompleteComponent.prototype, "headers", void 0);
    TableCompleteComponent = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Component"])({
      selector: 'app-table-complete',
      template: tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(
      /*! raw-loader!./table-complete.component.html */
      "./node_modules/raw-loader/dist/cjs.js!./src/app/content/components/tables/data-tables/table-complete/table-complete.component.html"))["default"],
      providers: [_country_service__WEBPACK_IMPORTED_MODULE_3__["CountryService"], _angular_common__WEBPACK_IMPORTED_MODULE_2__["DecimalPipe"]],
      styles: [tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(
      /*! ./table-complete.component.scss */
      "./src/app/content/components/tables/data-tables/table-complete/table-complete.component.scss"))["default"]]
    })], TableCompleteComponent);
    /***/
  },

  /***/
  "./src/app/content/components/tables/data-tables/table-filtering/table-filtering.component.scss":
  /*!******************************************************************************************************!*\
    !*** ./src/app/content/components/tables/data-tables/table-filtering/table-filtering.component.scss ***!
    \******************************************************************************************************/

  /*! exports provided: default */

  /***/
  function srcAppContentComponentsTablesDataTablesTableFilteringTableFilteringComponentScss(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony default export */


    __webpack_exports__["default"] = "\n/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IiIsImZpbGUiOiJzcmMvYXBwL2NvbnRlbnQvY29tcG9uZW50cy90YWJsZXMvZGF0YS10YWJsZXMvdGFibGUtZmlsdGVyaW5nL3RhYmxlLWZpbHRlcmluZy5jb21wb25lbnQuc2NzcyJ9 */";
    /***/
  },

  /***/
  "./src/app/content/components/tables/data-tables/table-filtering/table-filtering.component.ts":
  /*!****************************************************************************************************!*\
    !*** ./src/app/content/components/tables/data-tables/table-filtering/table-filtering.component.ts ***!
    \****************************************************************************************************/

  /*! exports provided: TableFilteringComponent */

  /***/
  function srcAppContentComponentsTablesDataTablesTableFilteringTableFilteringComponentTs(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony export (binding) */


    __webpack_require__.d(__webpack_exports__, "TableFilteringComponent", function () {
      return TableFilteringComponent;
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


    var _angular_forms__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(
    /*! @angular/forms */
    "./node_modules/@angular/forms/fesm2015/forms.js");
    /* harmony import */


    var rxjs_operators__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(
    /*! rxjs/operators */
    "./node_modules/rxjs/_esm2015/operators/index.js");

    var COUNTRIES = [{
      name: 'Russia',
      flag: 'f/f3/Flag_of_Russia.svg',
      area: 17075200,
      population: 146989754
    }, {
      name: 'Canada',
      flag: 'c/cf/Flag_of_Canada.svg',
      area: 9976140,
      population: 36624199
    }, {
      name: 'United States',
      flag: 'a/a4/Flag_of_the_United_States.svg',
      area: 9629091,
      population: 324459463
    }, {
      name: 'China',
      flag: 'f/fa/Flag_of_the_People%27s_Republic_of_China.svg',
      area: 9596960,
      population: 1409517397
    }];

    function search(text, pipe) {
      return COUNTRIES.filter(function (country) {
        var term = text.toLowerCase();
        return country.name.toLowerCase().includes(term) || pipe.transform(country.area).includes(term) || pipe.transform(country.population).includes(term);
      });
    }

    var TableFilteringComponent = function TableFilteringComponent(pipe) {
      _classCallCheck(this, TableFilteringComponent);

      this.filter = new _angular_forms__WEBPACK_IMPORTED_MODULE_3__["FormControl"]('');
      this.countries$ = this.filter.valueChanges.pipe(Object(rxjs_operators__WEBPACK_IMPORTED_MODULE_4__["startWith"])(''), Object(rxjs_operators__WEBPACK_IMPORTED_MODULE_4__["map"])(function (text) {
        return search(text, pipe);
      }));
    };

    TableFilteringComponent.ctorParameters = function () {
      return [{
        type: _angular_common__WEBPACK_IMPORTED_MODULE_2__["DecimalPipe"]
      }];
    };

    TableFilteringComponent = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Component"])({
      selector: 'app-table-filtering',
      template: tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(
      /*! raw-loader!./table-filtering.component.html */
      "./node_modules/raw-loader/dist/cjs.js!./src/app/content/components/tables/data-tables/table-filtering/table-filtering.component.html"))["default"],
      providers: [_angular_common__WEBPACK_IMPORTED_MODULE_2__["DecimalPipe"]],
      styles: [tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(
      /*! ./table-filtering.component.scss */
      "./src/app/content/components/tables/data-tables/table-filtering/table-filtering.component.scss"))["default"]]
    })], TableFilteringComponent);
    /***/
  },

  /***/
  "./src/app/content/components/tables/data-tables/table-pagination/table-pagination.component.scss":
  /*!********************************************************************************************************!*\
    !*** ./src/app/content/components/tables/data-tables/table-pagination/table-pagination.component.scss ***!
    \********************************************************************************************************/

  /*! exports provided: default */

  /***/
  function srcAppContentComponentsTablesDataTablesTablePaginationTablePaginationComponentScss(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony default export */


    __webpack_exports__["default"] = "\n/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IiIsImZpbGUiOiJzcmMvYXBwL2NvbnRlbnQvY29tcG9uZW50cy90YWJsZXMvZGF0YS10YWJsZXMvdGFibGUtcGFnaW5hdGlvbi90YWJsZS1wYWdpbmF0aW9uLmNvbXBvbmVudC5zY3NzIn0= */";
    /***/
  },

  /***/
  "./src/app/content/components/tables/data-tables/table-pagination/table-pagination.component.ts":
  /*!******************************************************************************************************!*\
    !*** ./src/app/content/components/tables/data-tables/table-pagination/table-pagination.component.ts ***!
    \******************************************************************************************************/

  /*! exports provided: TablePaginationComponent */

  /***/
  function srcAppContentComponentsTablesDataTablesTablePaginationTablePaginationComponentTs(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony export (binding) */


    __webpack_require__.d(__webpack_exports__, "TablePaginationComponent", function () {
      return TablePaginationComponent;
    });
    /* harmony import */


    var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(
    /*! tslib */
    "./node_modules/tslib/tslib.es6.js");
    /* harmony import */


    var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(
    /*! @angular/core */
    "./node_modules/@angular/core/fesm2015/core.js");

    var COUNTRIES = [{
      id: 1,
      name: 'Russia',
      flag: 'f/f3/Flag_of_Russia.svg',
      area: 17075200,
      population: 146989754
    }, {
      id: 2,
      name: 'France',
      flag: 'c/c3/Flag_of_France.svg',
      area: 640679,
      population: 64979548
    }, {
      id: 3,
      name: 'Germany',
      flag: 'b/ba/Flag_of_Germany.svg',
      area: 357114,
      population: 82114224
    }, {
      id: 4,
      name: 'Portugal',
      flag: '5/5c/Flag_of_Portugal.svg',
      area: 92090,
      population: 10329506
    }, {
      id: 5,
      name: 'Canada',
      flag: 'c/cf/Flag_of_Canada.svg',
      area: 9976140,
      population: 36624199
    }, {
      id: 6,
      name: 'Vietnam',
      flag: '2/21/Flag_of_Vietnam.svg',
      area: 331212,
      population: 95540800
    }, {
      id: 7,
      name: 'Brazil',
      flag: '0/05/Flag_of_Brazil.svg',
      area: 8515767,
      population: 209288278
    }, {
      id: 8,
      name: 'Mexico',
      flag: 'f/fc/Flag_of_Mexico.svg',
      area: 1964375,
      population: 129163276
    }, {
      id: 9,
      name: 'United States',
      flag: 'a/a4/Flag_of_the_United_States.svg',
      area: 9629091,
      population: 324459463
    }, {
      id: 10,
      name: 'India',
      flag: '4/41/Flag_of_India.svg',
      area: 3287263,
      population: 1324171354
    }, {
      id: 11,
      name: 'Indonesia',
      flag: '9/9f/Flag_of_Indonesia.svg',
      area: 1910931,
      population: 263991379
    }, {
      id: 12,
      name: 'Tuvalu',
      flag: '3/38/Flag_of_Tuvalu.svg',
      area: 26,
      population: 11097
    }, {
      id: 13,
      name: 'China',
      flag: 'f/fa/Flag_of_the_People%27s_Republic_of_China.svg',
      area: 9596960,
      population: 1409517397
    }];

    var TablePaginationComponent = /*#__PURE__*/function () {
      function TablePaginationComponent() {
        _classCallCheck(this, TablePaginationComponent);

        this.page = 1;
        this.pageSize = 4;
        this.collectionSize = COUNTRIES.length;
      }

      _createClass(TablePaginationComponent, [{
        key: "countries",
        get: function get() {
          return COUNTRIES // .map((country, i) => ({id: i + 1, ...country}))
          .slice((this.page - 1) * this.pageSize, (this.page - 1) * this.pageSize + this.pageSize);
        }
      }]);

      return TablePaginationComponent;
    }();

    TablePaginationComponent = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Component"])({
      selector: 'app-table-pagination',
      template: tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(
      /*! raw-loader!./table-pagination.component.html */
      "./node_modules/raw-loader/dist/cjs.js!./src/app/content/components/tables/data-tables/table-pagination/table-pagination.component.html"))["default"],
      styles: [tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(
      /*! ./table-pagination.component.scss */
      "./src/app/content/components/tables/data-tables/table-pagination/table-pagination.component.scss"))["default"]]
    })], TablePaginationComponent);
    /***/
  },

  /***/
  "./src/app/content/components/tables/data-tables/table-sortable/table-sortable.component.scss":
  /*!****************************************************************************************************!*\
    !*** ./src/app/content/components/tables/data-tables/table-sortable/table-sortable.component.scss ***!
    \****************************************************************************************************/

  /*! exports provided: default */

  /***/
  function srcAppContentComponentsTablesDataTablesTableSortableTableSortableComponentScss(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony default export */


    __webpack_exports__["default"] = "\n/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IiIsImZpbGUiOiJzcmMvYXBwL2NvbnRlbnQvY29tcG9uZW50cy90YWJsZXMvZGF0YS10YWJsZXMvdGFibGUtc29ydGFibGUvdGFibGUtc29ydGFibGUuY29tcG9uZW50LnNjc3MifQ== */";
    /***/
  },

  /***/
  "./src/app/content/components/tables/data-tables/table-sortable/table-sortable.component.ts":
  /*!**************************************************************************************************!*\
    !*** ./src/app/content/components/tables/data-tables/table-sortable/table-sortable.component.ts ***!
    \**************************************************************************************************/

  /*! exports provided: compare, TableSortableComponent */

  /***/
  function srcAppContentComponentsTablesDataTablesTableSortableTableSortableComponentTs(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony export (binding) */


    __webpack_require__.d(__webpack_exports__, "compare", function () {
      return compare;
    });
    /* harmony export (binding) */


    __webpack_require__.d(__webpack_exports__, "TableSortableComponent", function () {
      return TableSortableComponent;
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


    var _sortable_directive__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(
    /*! ./../sortable.directive */
    "./src/app/content/components/tables/data-tables/sortable.directive.ts");

    var COUNTRIES = [{
      id: 1,
      name: 'Russia',
      flag: 'f/f3/Flag_of_Russia.svg',
      area: 17075200,
      population: 146989754
    }, {
      id: 2,
      name: 'Canada',
      flag: 'c/cf/Flag_of_Canada.svg',
      area: 9976140,
      population: 36624199
    }, {
      id: 3,
      name: 'United States',
      flag: 'a/a4/Flag_of_the_United_States.svg',
      area: 9629091,
      population: 324459463
    }, {
      id: 4,
      name: 'China',
      flag: 'f/fa/Flag_of_the_People%27s_Republic_of_China.svg',
      area: 9596960,
      population: 1409517397
    }];

    var compare = function compare(v1, v2) {
      return v1 < v2 ? -1 : v1 > v2 ? 1 : 0;
    };

    var TableSortableComponent = /*#__PURE__*/function () {
      function TableSortableComponent() {
        _classCallCheck(this, TableSortableComponent);

        this.countries = COUNTRIES;
      }

      _createClass(TableSortableComponent, [{
        key: "onSort",
        value: function onSort(_ref2) {
          var column = _ref2.column,
              direction = _ref2.direction;
          // resetting other headers
          this.headers.forEach(function (header) {
            if (header.sortable !== column) {
              header.direction = '';
            }
          }); // sorting countries

          if (direction === '') {
            this.countries = COUNTRIES;
          } else {
            this.countries = [].concat(COUNTRIES).sort(function (a, b) {
              var res = compare(a[column], b[column]);
              return direction === 'asc' ? res : -res;
            });
          }
        }
      }]);

      return TableSortableComponent;
    }();

    tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["ViewChildren"])(_sortable_directive__WEBPACK_IMPORTED_MODULE_2__["SortableDirective"])], TableSortableComponent.prototype, "headers", void 0);
    TableSortableComponent = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Component"])({
      selector: 'app-table-sortable',
      template: tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(
      /*! raw-loader!./table-sortable.component.html */
      "./node_modules/raw-loader/dist/cjs.js!./src/app/content/components/tables/data-tables/table-sortable/table-sortable.component.html"))["default"],
      styles: [tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(
      /*! ./table-sortable.component.scss */
      "./src/app/content/components/tables/data-tables/table-sortable/table-sortable.component.scss"))["default"]]
    })], TableSortableComponent);
    /***/
  },

  /***/
  "./src/app/content/components/tables/tables.module.ts":
  /*!************************************************************!*\
    !*** ./src/app/content/components/tables/tables.module.ts ***!
    \************************************************************/

  /*! exports provided: TablesModule */

  /***/
  function srcAppContentComponentsTablesTablesModuleTs(module, __webpack_exports__, __webpack_require__) {
    "use strict";

    __webpack_require__.r(__webpack_exports__);
    /* harmony export (binding) */


    __webpack_require__.d(__webpack_exports__, "TablesModule", function () {
      return TablesModule;
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


    var _gaxon_modules__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(
    /*! @gaxon/modules */
    "./src/gaxon/modules/index.ts");
    /* harmony import */


    var _basic_tables_basic_tables_component__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(
    /*! ./basic-tables/basic-tables.component */
    "./src/app/content/components/tables/basic-tables/basic-tables.component.ts");
    /* harmony import */


    var _data_tables_data_tables_component__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(
    /*! ./data-tables/data-tables.component */
    "./src/app/content/components/tables/data-tables/data-tables.component.ts");
    /* harmony import */


    var _data_tables_table_sortable_table_sortable_component__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(
    /*! ./data-tables/table-sortable/table-sortable.component */
    "./src/app/content/components/tables/data-tables/table-sortable/table-sortable.component.ts");
    /* harmony import */


    var _data_tables_table_filtering_table_filtering_component__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(
    /*! ./data-tables/table-filtering/table-filtering.component */
    "./src/app/content/components/tables/data-tables/table-filtering/table-filtering.component.ts");
    /* harmony import */


    var _data_tables_table_pagination_table_pagination_component__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(
    /*! ./data-tables/table-pagination/table-pagination.component */
    "./src/app/content/components/tables/data-tables/table-pagination/table-pagination.component.ts");
    /* harmony import */


    var _data_tables_table_complete_table_complete_component__WEBPACK_IMPORTED_MODULE_10__ = __webpack_require__(
    /*! ./data-tables/table-complete/table-complete.component */
    "./src/app/content/components/tables/data-tables/table-complete/table-complete.component.ts");
    /* harmony import */


    var _data_tables_sortable_directive__WEBPACK_IMPORTED_MODULE_11__ = __webpack_require__(
    /*! ./data-tables/sortable.directive */
    "./src/app/content/components/tables/data-tables/sortable.directive.ts");

    var componentsRoutes = [{
      path: 'base-table',
      component: _basic_tables_basic_tables_component__WEBPACK_IMPORTED_MODULE_5__["BasicTablesComponent"]
    }, {
      path: 'data-table',
      component: _data_tables_data_tables_component__WEBPACK_IMPORTED_MODULE_6__["DataTablesComponent"]
    }];

    var TablesModule = function TablesModule() {
      _classCallCheck(this, TablesModule);
    };

    TablesModule = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["NgModule"])({
      imports: [_angular_common__WEBPACK_IMPORTED_MODULE_2__["CommonModule"], _gaxon_modules__WEBPACK_IMPORTED_MODULE_4__["SharedModule"], _angular_router__WEBPACK_IMPORTED_MODULE_3__["RouterModule"].forChild(componentsRoutes)],
      declarations: [_basic_tables_basic_tables_component__WEBPACK_IMPORTED_MODULE_5__["BasicTablesComponent"], _data_tables_data_tables_component__WEBPACK_IMPORTED_MODULE_6__["DataTablesComponent"], _data_tables_table_sortable_table_sortable_component__WEBPACK_IMPORTED_MODULE_7__["TableSortableComponent"], _data_tables_table_filtering_table_filtering_component__WEBPACK_IMPORTED_MODULE_8__["TableFilteringComponent"], _data_tables_table_pagination_table_pagination_component__WEBPACK_IMPORTED_MODULE_9__["TablePaginationComponent"], _data_tables_table_complete_table_complete_component__WEBPACK_IMPORTED_MODULE_10__["TableCompleteComponent"], _data_tables_sortable_directive__WEBPACK_IMPORTED_MODULE_11__["SortableDirective"]]
    })], TablesModule);
    /***/
  }
}]);
//# sourceMappingURL=tables-tables-module-es5.js.map