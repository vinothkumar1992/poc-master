(window["webpackJsonp"] = window["webpackJsonp"] || []).push([["contacts-contacts-module"],{

/***/ "./node_modules/raw-loader/dist/cjs.js!./src/app/content/apps/contacts/contacts-header/contacts-header.component.html":
/*!****************************************************************************************************************************!*\
  !*** ./node_modules/raw-loader/dist/cjs.js!./src/app/content/apps/contacts/contacts-header/contacts-header.component.html ***!
  \****************************************************************************************************************************/
/*! exports provided: default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony default export */ __webpack_exports__["default"] = ("<contact-search-bar></contact-search-bar>\r\n\r\n<!-- Dropdown -->\r\n<div class=\"ml-auto dt-notification\" ngbDropdown placement=\"bottom-right\">\r\n\r\n  <!-- Dropdown Button -->\r\n  <a href=\"javascript:void(0)\" class=\"no-arrow\" ngbDropdownToggle>\r\n    <gx-icon name=\"notification2\" [fw]=\"true\" size=\"1x\" class=\"dt-icon-alert\"></gx-icon>\r\n  </a>\r\n  <!-- /dropdown button -->\r\n\r\n  <!-- Dropdown Menu -->\r\n  <div ngbDropdownMenu class=\"dropdown-menu-media\">\r\n    <!-- Dropdown Menu Header -->\r\n    <div class=\"dropdown-menu-header\">\r\n      <h4 class=\"title\">Notifications (9)</h4>\r\n\r\n      <div class=\"ml-auto action-area\">\r\n        <a href=\"javascript:void(0)\">Mark All Read</a> <a class=\"ml-2\" href=\"javascript:void(0)\">\r\n        <gx-icon name=\"settings\" size=\"lg\" class=\"text-light-gray\"></gx-icon>\r\n      </a>\r\n      </div>\r\n    </div>\r\n    <!-- /dropdown menu header -->\r\n\r\n    <!-- Dropdown Menu Body -->\r\n    <div class=\"dropdown-menu-body position-relative\" perfectScrollbar>\r\n\r\n      <div class=\"h-auto\">\r\n        <!-- Media -->\r\n        <a href=\"javascript:void(0)\" class=\"media\" *ngFor=\"let notification of notifications\">\r\n\r\n          <!-- Avatar -->\r\n          <gx-avatar [user]=\"notification\" [charLength]=\"0\" class=\"mr-3\"></gx-avatar>\r\n          <!-- avatar -->\r\n\r\n          <!-- Media Body -->\r\n          <span class=\"media-body\">\r\n            <span class=\"message\" [innerHTML]=\"notification.message\"></span>\r\n            <span class=\"meta-date\">\r\n              <gx-icon *ngIf=\"notification.icon.name\"\r\n                       [name]=\"notification.icon.name\"\r\n                       [ngClass]=\"notification.icon.color\"\r\n                       [fw]=\"true\" class=\"mr-1\"></gx-icon>\r\n              {{notification.time}}\r\n            </span>\r\n          </span>\r\n          <!-- /media body -->\r\n\r\n        </a>\r\n        <!-- /media -->\r\n      </div>\r\n    </div>\r\n\r\n    <!-- Dropdown Menu Footer -->\r\n    <div class=\"dropdown-menu-footer\">\r\n      <a href=\"javascript:void(0)\" class=\"card-link\"> See All\r\n        <gx-icon name=\"arrow-right\" [fw]=\"true\"></gx-icon></a>\r\n    </div>\r\n    <!-- /dropdown menu footer -->\r\n  </div>\r\n  <!-- /dropdown menu -->\r\n\r\n</div>\r\n<!-- /dropdown -->\r\n");

/***/ }),

/***/ "./node_modules/raw-loader/dist/cjs.js!./src/app/content/apps/contacts/contacts-header/search-bar/search-bar.component.html":
/*!**********************************************************************************************************************************!*\
  !*** ./node_modules/raw-loader/dist/cjs.js!./src/app/content/apps/contacts/contacts-header/search-bar/search-bar.component.html ***!
  \**********************************************************************************************************************************/
/*! exports provided: default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony default export */ __webpack_exports__["default"] = ("<form>\r\n  <input type=\"search\"\r\n         class=\"form-control border-0 shadow-none bg-focus form-control-sm\"\r\n         [formControl]=\"searchQuery\"\r\n         placeholder=\"Search in contacts...\">\r\n  <span class=\"search-icon\"><gx-icon name=\"search\" size=\"lg\"></gx-icon></span>\r\n</form>\r\n");

/***/ }),

/***/ "./node_modules/raw-loader/dist/cjs.js!./src/app/content/apps/contacts/contacts-list/contact-list-header/contact-list-header.component.html":
/*!**************************************************************************************************************************************************!*\
  !*** ./node_modules/raw-loader/dist/cjs.js!./src/app/content/apps/contacts/contacts-list/contact-list-header/contact-list-header.component.html ***!
  \**************************************************************************************************************************************************/
/*! exports provided: default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony default export */ __webpack_exports__["default"] = ("<!-- Action Bar -->\r\n<div class=\"d-flex flex-wrap\">\r\n  <!-- Dropdown -->\r\n  <div ngbDropdown class=\"mr-auto\">\r\n\r\n    <!-- Checkbox -->\r\n    <gx-checkbox class=\"mr-3\"\r\n                 [gxModel]=\"hasSelectedContacts\"\r\n                 [indeterminate]=\"isIndeterminate\"\r\n                 (gxInputChange)=\"toggleSelectAll()\"></gx-checkbox>\r\n    <!-- /checkbox -->\r\n\r\n    <!-- Dropdown Button -->\r\n    <a href=\"javascript:void(0)\" class=\"text-dark\" ngbDropdownToggle> {{selected}} </a>\r\n    <!-- /dropdown button -->\r\n\r\n    <!-- Dropdown Menu -->\r\n    <div ngbDropdownMenu>\r\n      <a class=\"dropdown-item\" href=\"javascript:void(0)\" (click)=\"selectContacts()\">All</a>\r\n      <a class=\"dropdown-item\" href=\"javascript:void(0)\" (click)=\"deselectContacts()\">None</a>\r\n      <a class=\"dropdown-item\" href=\"javascript:void(0)\"\r\n         (click)=\"selectContacts('Starred', 'starred', true)\">Starred</a>\r\n      <a class=\"dropdown-item\" href=\"javascript:void(0)\"\r\n         (click)=\"selectContacts('Unstarred', 'starred', false)\">Unstarred</a>\r\n      <a class=\"dropdown-item\" href=\"javascript:void(0)\"\r\n         (click)=\"selectContacts('Favourite', 'favourite', true)\">Favourite</a>\r\n      <a class=\"dropdown-item\" href=\"javascript:void(0)\"\r\n         (click)=\"selectContacts('Unfavourite', 'favourite', false)\">Unfavourite</a>\r\n    </div>\r\n    <!-- /dropdown menu -->\r\n\r\n  </div>\r\n  <!-- /dropdown -->\r\n\r\n  <!-- Button Group -->\r\n  <div *ngIf=\"hasSelectedContacts\" class=\"btn-group btn-group-sm mr-4 mr-lg-7 mb-1\" role=\"group\">\r\n    <button type=\"button\" class=\"btn btn-outline-dark\" (click)=\"setPropertyOnSelectedContacts('favourite', true)\">\r\n      <gx-icon name=\"heart-o\" [fw]=\"true\" size=\"lg\"></gx-icon>\r\n    </button>\r\n    <button type=\"button\" class=\"btn btn-outline-dark\" (click)=\"setPropertyOnSelectedContacts('starred', true)\">\r\n      <gx-icon name=\"star-fill\" [fw]=\"true\" size=\"lg\"></gx-icon>\r\n    </button>\r\n    <button type=\"button\" class=\"btn btn-outline-dark\" (click)=\"setPropertyOnSelectedContacts('suggested', true)\">\r\n      <gx-icon name=\"contacts\" [fw]=\"true\" size=\"lg\"></gx-icon>\r\n    </button>\r\n  </div>\r\n  <!-- /button group -->\r\n\r\n  <!-- Dropdown -->\r\n  <div *ngIf=\"hasSelectedContacts\" ngbDropdown placement=\"bottom-right\">\r\n\r\n    <!-- Dropdown Button -->\r\n    <a href=\"javascript:void(0)\" class=\"btn btn-outline-dark btn-sm\" ngbDropdownToggle>\r\n      More\r\n    </a>\r\n    <!-- /dropdown button -->\r\n\r\n    <!-- Dropdown Menu -->\r\n    <div ngbDropdownMenu>\r\n      <a class=\"dropdown-item\" href=\"javascript:void(0)\" (click)=\"setPropertyOnSelectedContacts('favourite', false)\">Unfavourite</a>\r\n      <a class=\"dropdown-item\" href=\"javascript:void(0)\" (click)=\"setPropertyOnSelectedContacts('favourite', true)\">Favourite</a>\r\n      <a class=\"dropdown-item\" href=\"javascript:void(0)\" (click)=\"setPropertyOnSelectedContacts('suggested', true)\">Add\r\n        in Suggestions</a>\r\n      <a class=\"dropdown-item\" href=\"javascript:void(0)\" (click)=\"setPropertyOnSelectedContacts('starred', true)\">Starred</a>\r\n      <a class=\"dropdown-item\" href=\"javascript:void(0)\" (click)=\"setPropertyOnSelectedContacts('starred', false)\">Unstarred</a>\r\n    </div>\r\n    <!-- /dropdown menu -->\r\n\r\n  </div>\r\n  <!-- /dropdown -->\r\n</div>\r\n<!-- /action bar -->\r\n");

/***/ }),

/***/ "./node_modules/raw-loader/dist/cjs.js!./src/app/content/apps/contacts/contacts-list/contact-list-item/contact-list-item.component.html":
/*!**********************************************************************************************************************************************!*\
  !*** ./node_modules/raw-loader/dist/cjs.js!./src/app/content/apps/contacts/contacts-list/contact-list-item/contact-list-item.component.html ***!
  \**********************************************************************************************************************************************/
/*! exports provided: default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony default export */ __webpack_exports__["default"] = ("<!-- Checkbox -->\r\n<gx-checkbox class=\"mr-5\" [gxModel]=\"selected\" (gxInputChange)=\"onSelectedChange($event)\"></gx-checkbox>\r\n<!-- /checkbox -->\r\n\r\n<gx-star class=\"mr-5\" [gxModel]=\"contact.starred\" (gxInputChange)=\"onToggleStarred($event)\"></gx-star>\r\n\r\n<gx-avatar [user]=\"contact\" class=\"mr-4\"></gx-avatar>\r\n\r\n<!-- Module Content -->\r\n<div class=\"dt-module__list-item-content\">\r\n  <div class=\"user-detail\">\r\n    <span class=\"user-name\">{{contact.first_name}}</span>\r\n    <span class=\"dt-separator-v\">&nbsp;</span>\r\n    <span class=\"designation\">{{contact.designation | htmlToPlainText | slice:0:30}}</span>\r\n  </div>\r\n  <div class=\"text-light-gray\">\r\n    <span class=\"d-inline-block mr-3\">{{contact.email}}</span>\r\n    <span class=\"d-inline-block\">{{contact.phone}}</span>\r\n  </div>\r\n</div>\r\n<!-- /module content -->\r\n\r\n<!-- Module Info -->\r\n<div class=\"dt-module__list-item-info\">\r\n  <!-- Dropdown -->\r\n  <div ngbDropdown placement=\"bottom-right\">\r\n    <a class=\"no-arrow\" href=\"javascript:void(0)\" ngbDropdownToggle>\r\n      <gx-icon name=\"vertical-more\" [fw]=\"true\" size=\"xl\" class=\"text-light-gray\"></gx-icon>\r\n    </a>\r\n\r\n    <div ngbDropdownMenu>\r\n      <a class=\"dropdown-item\" href=\"javascript:void(0)\"  (click)=\"editContact(contact)\">\r\n        <gx-icon name=\"editors\" [fw]=\"true\"></gx-icon>\r\n        Edit\r\n      </a>\r\n      <a class=\"dropdown-item\" href=\"javascript:void(0)\" (click)=\"deleteContact(contact)\">\r\n        <gx-icon name=\"trash\" [fw]=\"true\"></gx-icon>\r\n        Delete\r\n      </a>\r\n      <a class=\"dropdown-item\" href=\"javascript:void(0)\" (click)=\"onFavouriteContact()\">\r\n        <gx-icon name=\"heart-o\" [fw]=\"true\"></gx-icon>\r\n        Favourite List\r\n      </a>\r\n    </div>\r\n  </div>\r\n  <!-- /dropdown -->\r\n</div>\r\n<!-- /module info -->\r\n");

/***/ }),

/***/ "./node_modules/raw-loader/dist/cjs.js!./src/app/content/apps/contacts/contacts-list/contacts-list.component.html":
/*!************************************************************************************************************************!*\
  !*** ./node_modules/raw-loader/dist/cjs.js!./src/app/content/apps/contacts/contacts-list/contacts-list.component.html ***!
  \************************************************************************************************************************/
/*! exports provided: default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony default export */ __webpack_exports__["default"] = ("<!-- Module Content Inner -->\r\n<div class=\"dt-module__content-inner\">\r\n  <app-contact-list-header></app-contact-list-header>\r\n\r\n  <!-- Module List -->\r\n  <div class=\"dt-module__list\">\r\n\r\n    <div *ngIf=\"contacts.length == 0\" class=\"dt-module__list-item\">\r\n      <p class=\"empty-result\">No record found.</p>\r\n    </div>\r\n\r\n    <!-- Module Item -->\r\n    <app-contact-list-item *ngFor=\"let contact of contacts | filter: searchQuery\"\r\n                           [contact]=\"contact\"></app-contact-list-item>\r\n    <!-- /Module Item -->\r\n\r\n  </div>\r\n  <!-- /Module List -->\r\n</div>\r\n");

/***/ }),

/***/ "./node_modules/raw-loader/dist/cjs.js!./src/app/content/apps/contacts/contacts-sidebar/contacts-sidebar.component.html":
/*!******************************************************************************************************************************!*\
  !*** ./node_modules/raw-loader/dist/cjs.js!./src/app/content/apps/contacts/contacts-sidebar/contacts-sidebar.component.html ***!
  \******************************************************************************************************************************/
/*! exports provided: default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony default export */ __webpack_exports__["default"] = ("<!-- Sidebar Header -->\r\n<div class=\"dt-module__sidebar-header border-bottom\">\r\n  <div class=\"d-none d-md-flex align-items-center\">\r\n    <gx-icon name=\"user-account\" size=\"1x\" class=\"mr-3 text-dark\"></gx-icon>\r\n    <span class=\"h3 mb-0\">Contacts</span>\r\n  </div>\r\n\r\n  <!-- App Quick Menu -->\r\n  <div class=\"quick-menu-list d-md-none\">\r\n    <a *ngIf=\"hiddenPageHeader\"\r\n       href=\"javascript:void(0)\"\r\n       class=\"quick-menu mr-auto d-md-none\"\r\n       [ngClass]=\"{'active':settings.activeNavDrawer}\"\r\n       (click)=\"toggleNavigation($event)\">\r\n      <gx-icon name=\"menu\"></gx-icon>\r\n    </a>\r\n\r\n    <!-- Search Box -->\r\n    <form class=\"search-box d-md-none\" *ngIf=\"!hiddenPageHeader\">\r\n      <input class=\"form-control\" type=\"search\" id=\"address\" name=\"address\" placeholder=\"Search in app...\">\r\n      <button type=\"submit\" class=\"search-icon\">\r\n        <gx-icon name=\"search\" size=\"lg\"></gx-icon>\r\n      </button>\r\n    </form>\r\n    <!-- /search box -->\r\n\r\n    <a href=\"javascript:void(0)\"\r\n       class=\"quick-menu\"\r\n       [ngClass]=\"{'active':activeMenu == 'search', 'd-none d-md-block': !hiddenPageHeader}\"\r\n       (click)=\"onClickMenu($event, 'search')\">\r\n      <gx-icon name=\"search\"></gx-icon>\r\n    </a>\r\n\r\n    <a href=\"javascript:void(0)\" class=\"quick-menu\" (click)=\"onClickNewContact()\">\r\n      <gx-icon name=\"contacts\"></gx-icon>\r\n    </a>\r\n\r\n    <a href=\"javascript:void(0)\"\r\n       class=\"quick-menu\"\r\n       [ngClass]=\"{'active':activeMenu == 'notifications'}\"\r\n       (click)=\"onClickMenu($event, 'notifications')\">\r\n      <gx-icon name=\"notification2\"></gx-icon>\r\n    </a>\r\n\r\n    <a href=\"javascript:void(0)\"\r\n       class=\"quick-menu d-md-none\"\r\n       [ngClass]=\"{'active':isOpen}\"\r\n       (click)=\"toggleOpen($event)\">\r\n      <gx-icon name=\"{{hiddenPageHeader? 'vertical-more': 'menu'}}\"></gx-icon>\r\n    </a>\r\n  </div>\r\n  <!-- /app quick menu -->\r\n</div>\r\n<!-- /sidebar header -->\r\n\r\n<!-- Sidebar Menu -->\r\n<div class=\"dt-module__sidebar-content position-relative\" [@collapseAppSidebar]=\"isOpen\" perfectScrollbar>\r\n  <!-- Sidebar Content Inner -->\r\n  <div class=\"dt-module__sidebar-content-inner\">\r\n    <!-- Button -->\r\n    <div class=\"action-area mt-5 mb-7 d-none d-md-block\">\r\n      <a href=\"javascript:void(0)\"\r\n         class=\"btn btn-shadow btn-default compose-btn btn-rounded\"\r\n         (click)=\"onClickNewContact()\">\r\n        <gx-icon name=\"contacts\" class=\"mr-2\" [fw]=\"true\"></gx-icon>Add Contact\r\n      </a>\r\n    </div>\r\n    <!-- /button -->\r\n\r\n    <!-- Sidebar Navigation -->\r\n    <ul class=\"dt-module-side-nav\">\r\n\r\n      <!-- Menu Header -->\r\n      <li class=\"dt-module-side-nav__header\">\r\n        <span class=\"dt-module-side-nav__text\">Contacts</span>\r\n      </li>\r\n      <!-- /menu header -->\r\n\r\n      <!-- Menu Item -->\r\n      <li class=\"dt-module-side-nav__item\" routerLinkActive=\"active\">\r\n        <a class=\"dt-module-side-nav__link\" [routerLink]=\"'/' + settings.layout + '/apps/contacts/all'\">\r\n          <gx-icon name=\"contacts\" size=\"lg\" [fw]=\"true\"></gx-icon>\r\n          <span class=\"dt-module-side-nav__text\">All Contacts</span>\r\n        </a>\r\n      </li>\r\n      <!-- /menu item -->\r\n\r\n      <!-- Menu Item -->\r\n      <li class=\"dt-module-side-nav__item\" routerLinkActive=\"active\">\r\n        <a class=\"dt-module-side-nav__link\" [routerLink]=\"'/' + settings.layout + '/apps/contacts/favourite'\">\r\n          <gx-icon name=\"heart-o\" size=\"lg\" [fw]=\"true\"></gx-icon>\r\n          <span class=\"dt-module-side-nav__text\">Favourite List</span>\r\n        </a>\r\n      </li>\r\n      <!-- /menu item -->\r\n\r\n      <!-- Menu Item -->\r\n      <li class=\"dt-module-side-nav__item\" routerLinkActive=\"active\">\r\n        <a class=\"dt-module-side-nav__link\" [routerLink]=\"'/' + settings.layout + '/apps/contacts/suggested'\">\r\n          <gx-icon name=\"contacts\" size=\"lg\" [fw]=\"true\"></gx-icon>\r\n          <span class=\"dt-module-side-nav__text\">Suggestions</span>\r\n        </a>\r\n      </li>\r\n      <!-- /menu item -->\r\n\r\n      <!-- Menu Item -->\r\n      <li class=\"dt-module-side-nav__item\" routerLinkActive=\"active\">\r\n        <a class=\"dt-module-side-nav__link\" [routerLink]=\"'/' + settings.layout + '/apps/contacts/starred'\">\r\n          <gx-icon name=\"star-o\" size=\"lg\" [fw]=\"true\"></gx-icon>\r\n          <span class=\"dt-module-side-nav__text\">Starred contacts</span>\r\n        </a>\r\n      </li>\r\n      <!-- /menu item -->\r\n\r\n      <!-- Menu Item -->\r\n      <li class=\"dt-module-side-nav__item\">\r\n        <a href=\"javascript:void(0)\" class=\"dt-module-side-nav__link\">\r\n          <gx-icon name=\"settings\" size=\"lg\" [fw]=\"true\"></gx-icon>\r\n          <span class=\"dt-module-side-nav__text\">Settings</span>\r\n        </a>\r\n      </li>\r\n      <!-- /menu item -->\r\n\r\n    </ul>\r\n    <!-- /sidebar navigation -->\r\n  </div>\r\n  <!-- /sidebar content inner -->\r\n</div>\r\n<!-- /sidebar Menu -->\r\n");

/***/ }),

/***/ "./node_modules/raw-loader/dist/cjs.js!./src/app/content/apps/contacts/contacts.component.html":
/*!*****************************************************************************************************!*\
  !*** ./node_modules/raw-loader/dist/cjs.js!./src/app/content/apps/contacts/contacts.component.html ***!
  \*****************************************************************************************************/
/*! exports provided: default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony default export */ __webpack_exports__["default"] = ("<!-- Grid -->\r\n<div class=\"row\">\r\n\r\n  <!-- Grid Item -->\r\n  <div class=\"col-xl-12\">\r\n\r\n    <!-- Module -->\r\n    <div class=\"dt-module\">\r\n\r\n      <!-- Module Sidebar -->\r\n      <app-contacts-sidebar></app-contacts-sidebar>\r\n      <!-- /Module Sidebar -->\r\n\r\n      <!-- Module Container -->\r\n      <div class=\"dt-module__container\">\r\n\r\n        <!-- Module Header -->\r\n        <app-contacts-header></app-contacts-header>\r\n        <!-- /Module Header -->\r\n\r\n        <!-- Module Content -->\r\n        <app-contacts-list perfectScrollbar></app-contacts-list>\r\n        <!-- Module Content -->\r\n\r\n      </div>\r\n      <!-- /Module Container -->\r\n\r\n      <!-- Module Drawer -->\r\n      <contact-drawer [ngClass]=\"{'open':drawer.isOpen}\"></contact-drawer>\r\n      <div class=\"dt-backdrop\" *ngIf=\"drawer.isOpen\" (click)=\"closeDrawer($event)\"></div>\r\n      <!-- Module Drawer -->\r\n\r\n    </div>\r\n    <!-- /Module -->\r\n\r\n  </div>\r\n  <!-- /Grid Item -->\r\n\r\n</div>\r\n<!-- /Grid -->\r\n");

/***/ }),

/***/ "./node_modules/raw-loader/dist/cjs.js!./src/app/content/apps/contacts/drawer/drawer.component.html":
/*!**********************************************************************************************************!*\
  !*** ./node_modules/raw-loader/dist/cjs.js!./src/app/content/apps/contacts/drawer/drawer.component.html ***!
  \**********************************************************************************************************/
/*! exports provided: default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony default export */ __webpack_exports__["default"] = ("<!-- Module Drawer Inner -->\r\n<div class=\"dt-module__drawer-inner\">\r\n\r\n  <div class=\"dt-module__drawer-header\">\r\n    <a href=\"javascript:void(0)\" class=\"d-inline-block\" (click)=\"closeDrawer($event)\">\r\n      <gx-icon name=\"arrow-long-left\" size=\"2x\" class=\"mr-3\"></gx-icon>\r\n      <span class=\"align-middle\">Back</span>\r\n    </a>\r\n  </div>\r\n\r\n  <!-- Actionable Drawer -->\r\n  <div class=\"dt-actionable-drawer\">\r\n\r\n    <!-- Menu Bar -->\r\n    <div class=\"dt-action-bar\">\r\n\r\n      <!-- Action Menu -->\r\n      <ul class=\"dt-action-nav\">\r\n\r\n        <!-- Menu Item -->\r\n        <li class=\"dt-action-nav__item\" [ngClass]=\"{'active':activeMenu == 'notifications'}\">\r\n          <a href=\"javascript:void(0)\"\r\n             class=\"dt-action-nav__link\"\r\n             (click)=\"onSelectMenu($event, 'notifications')\">\r\n            <gx-icon name=\"notification2\" [fw]=\"true\"></gx-icon>\r\n          </a>\r\n        </li>\r\n        <!-- /menu item -->\r\n\r\n        <!-- Menu Item -->\r\n        <li class=\"dt-action-nav__item\" [ngClass]=\"{'active':activeMenu == 'search'}\">\r\n          <a href=\"javascript:void(0)\"\r\n             class=\"dt-action-nav__link\"\r\n             (click)=\"onSelectMenu($event, 'search')\">\r\n            <gx-icon name=\"search\" [fw]=\"true\"></gx-icon>\r\n          </a>\r\n        </li>\r\n        <!-- /menu item -->\r\n\r\n      </ul>\r\n      <!-- /action menu -->\r\n\r\n    </div>\r\n    <!-- /menu bar -->\r\n\r\n    <!-- Drawer Content Area -->\r\n    <div class=\"dt-action-content-wrapper\">\r\n\r\n      <gx-loader [ngClass]=\"{'active':showLoader}\"></gx-loader>\r\n      <div class=\"dt-action-content fade position-relative\" [ngClass]=\"{'show':!showLoader && activeMenu == 'search'}\" perfectScrollbar>\r\n        <todo-search></todo-search>\r\n      </div>\r\n      <div class=\"dt-action-content fade position-relative\" [ngClass]=\"{'show':!showLoader && activeMenu == 'notifications'}\" perfectScrollbar>\r\n        <todo-notifications></todo-notifications>\r\n      </div>\r\n\r\n    </div>\r\n    <!-- /drawer content area -->\r\n\r\n  </div>\r\n  <!-- /actionable drawer -->\r\n\r\n</div>\r\n<!-- /module drawer inner -->\r\n");

/***/ }),

/***/ "./node_modules/raw-loader/dist/cjs.js!./src/app/content/apps/contacts/drawer/notifications/notifications.component.html":
/*!*******************************************************************************************************************************!*\
  !*** ./node_modules/raw-loader/dist/cjs.js!./src/app/content/apps/contacts/drawer/notifications/notifications.component.html ***!
  \*******************************************************************************************************************************/
/*! exports provided: default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony default export */ __webpack_exports__["default"] = ("<div class=\"d-flex align-items-center mb-6\">\r\n  <h5 class=\"mb-0\">Notifications ({{notifications.length}})</h5>\r\n\r\n  <div class=\"ml-auto action-area\">\r\n    <a href=\"javascript:void(0)\">Mark All Read</a>\r\n\r\n    <!-- Dropdown -->\r\n    <span class=\"ml-2 d-inline-block\" ngbDropdown placement=\"bottom-right\">\r\n      <a class=\"no-arrow\" href=\"javascript:void(0)\" ngbDropdownToggle>\r\n        <gx-icon name=\"settings\" size=\"lg\" class=\"text-light-gray\"></gx-icon>\r\n      </a>\r\n\r\n      <span ngbDropdownMenu>\r\n        <a class=\"dropdown-item\" href=\"javascript:void(0)\">\r\n          <gx-icon name=\"editors\" [fw]=\"true\"></gx-icon>\r\n          Edit\r\n        </a>\r\n        <a class=\"dropdown-item\" href=\"javascript:void(0)\">\r\n          <gx-icon name=\"contacts\" [fw]=\"true\"></gx-icon>\r\n          Contact List</a>\r\n        <a class=\"dropdown-item\" href=\"javascript:void(0)\">\r\n          <gx-icon name=\"heart\" [fw]=\"true\"></gx-icon>\r\n          Favourite List\r\n        </a>\r\n      </span>\r\n    </span>\r\n    <!-- /dropdown -->\r\n  </div>\r\n</div>\r\n\r\n<!-- Media -->\r\n<a href=\"javascript:void(0)\" class=\"media\" *ngFor=\"let notification of notifications\">\r\n\r\n  <!-- Avatar -->\r\n  <gx-avatar [user]=\"notification\" [charLength]=\"0\" class=\"mr-3\"></gx-avatar>\r\n  <!-- avatar -->\r\n\r\n  <!-- Media Body -->\r\n  <span class=\"media-body\">\r\n    <span class=\"message\" [innerHTML]=\"notification.message\"></span>\r\n    <span class=\"meta-date\">\r\n      <gx-icon *ngIf=\"notification.icon.name\"\r\n               [name]=\"notification.icon.name\"\r\n               [ngClass]=\"notification.icon.color\"\r\n               [fw]=\"true\" class=\"mr-1\"></gx-icon>\r\n      {{notification.time}}\r\n    </span>\r\n  </span>\r\n  <!-- /media body -->\r\n\r\n</a>\r\n<!-- /media -->\r\n");

/***/ }),

/***/ "./node_modules/raw-loader/dist/cjs.js!./src/app/content/apps/contacts/drawer/search/search.component.html":
/*!*****************************************************************************************************************!*\
  !*** ./node_modules/raw-loader/dist/cjs.js!./src/app/content/apps/contacts/drawer/search/search.component.html ***!
  \*****************************************************************************************************************/
/*! exports provided: default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony default export */ __webpack_exports__["default"] = ("<!-- Search Box -->\r\n<form class=\"search-box right-side-icon mw-100 mb-8\">\r\n  <input type=\"search\"\r\n         class=\"form-control form-control-lg\"\r\n         [formControl]=\"searchQuery\"\r\n         placeholder=\"Search in mails...\">\r\n  <button type=\"submit\" class=\"search-icon\"><gx-icon name=\"search\" size=\"lg\"></gx-icon></button>\r\n</form>\r\n<!-- /search box -->\r\n\r\n<h4>Recently Viewed</h4>\r\n\r\n<!-- Search Result-->\r\n<div class=\"search-result\" *ngFor=\"let message of messages | filter: searchText\">\r\n  <a href=\"javascript:void(0)\" class=\"dt-avatar-wrapper\">\r\n    <!-- Avatar -->\r\n    <gx-avatar [user]=\"message\" [charLength]=\"0\" class=\"font-italic\"></gx-avatar>\r\n    <!-- /avatar -->\r\n\r\n    <!-- Avatar Info -->\r\n    <span class=\"dt-avatar-info\">\r\n      <span class=\"dt-avatar-name\">{{message.name}}</span>\r\n      <span class=\"f-12\" [innerHTML]=\"message.message\"></span>\r\n    </span>\r\n    <!-- /avatar info -->\r\n  </a>\r\n</div>\r\n<!-- /search result-->\r\n");

/***/ }),

/***/ "./node_modules/raw-loader/dist/cjs.js!./src/app/content/apps/contacts/edit-contact/edit-contact.component.html":
/*!**********************************************************************************************************************!*\
  !*** ./node_modules/raw-loader/dist/cjs.js!./src/app/content/apps/contacts/edit-contact/edit-contact.component.html ***!
  \**********************************************************************************************************************/
/*! exports provided: default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony default export */ __webpack_exports__["default"] = ("<div class=\"modal-header\">\r\n  <h4 class=\"modal-title\">{{title}}</h4>\r\n  <button type=\"button\" class=\"close\" aria-label=\"Close\" (click)=\"activeModal.dismiss('Cross click')\">\r\n    <gx-icon name=\"remove\" [fw]=\"true\"></gx-icon>\r\n  </button>\r\n</div>\r\n<div class=\"modal-body\">\r\n  <div class=\"row no-gutters\">\r\n    <div class=\"col-lg-4 text-center text-center order-lg-2\">\r\n      <img *ngIf=\"contact.thumb\" class=\"ml-lg-3 mb-4 mb-lg-0 dt-avatar size-120\" [src]=\"contact.thumb\" alt=\"Avatar\">\r\n      <img *ngIf=\"!contact.thumb\" class=\"ml-lg-3 mb-4 mb-lg-0 dt-avatar size-120\" src=\"assets/images/placeholder.jpg\"\r\n           alt=\"Avatar\">\r\n\r\n      <div class=\"form-group col-lg-12 mt-4\">\r\n        <div class=\"input-group\">\r\n          <div class=\"custom-file\">\r\n            <input type=\"file\"\r\n                   class=\"custom-file-input\"\r\n                   id=\"imageUrl\"\r\n                   name=\"imageUrl\"\r\n                   accept=\"image/*\"\r\n                   (change)=\"handleInputChange($event)\">\r\n            <label class=\"custom-file-label text-left\" for=\"imageUrl\">{{filename}}</label>\r\n          </div>\r\n        </div>\r\n      </div>\r\n    </div>\r\n\r\n    <div class=\"col-lg-8 order-lg-1\">\r\n      <form #contactForm=\"ngForm\" (ngSubmit)=\"onFormSubmit()\">\r\n        <div class=\"form-group\">\r\n          <input type=\"text\"\r\n                 class=\"form-control\"\r\n                 placeholder=\"First Name\"\r\n                 name=\"fisrt_name\"\r\n                 [(ngModel)]=\"contact.first_name\"\r\n                 required>\r\n        </div>\r\n\r\n        <div class=\"form-group\">\r\n          <input type=\"text\"\r\n                 class=\"form-control\"\r\n                 placeholder=\"Last Name\"\r\n                 name=\"last_name\"\r\n                 [(ngModel)]=\"contact.last_name\"\r\n                 required>\r\n        </div>\r\n\r\n        <div class=\"form-group\">\r\n          <input type=\"email\"\r\n                 class=\"form-control\"\r\n                 placeholder=\"Email\"\r\n                 name=\"email\"\r\n                 [(ngModel)]=\"contact.email\"\r\n                 required>\r\n        </div>\r\n\r\n        <div class=\"form-group\">\r\n          <input type=\"text\"\r\n                 class=\"form-control\"\r\n                 placeholder=\"Phone\"\r\n                 name=\"phone\"\r\n                 [(ngModel)]=\"contact.phone\"\r\n                 required>\r\n        </div>\r\n\r\n        <div class=\"form-group\">\r\n          <input type=\"text\"\r\n                 class=\"form-control\"\r\n                 placeholder=\"Designation\"\r\n                 name=\"designation\"\r\n                 [(ngModel)]=\"contact.designation\"\r\n                 required>\r\n        </div>\r\n\r\n        <div class=\"form-group\">\r\n          <textarea type=\"text\"\r\n                    class=\"form-control\"\r\n                    placeholder=\"Designation\"\r\n                    name=\"bio\"\r\n                    [(ngModel)]=\"contact.bio\"></textarea>\r\n        </div>\r\n      </form>\r\n    </div>\r\n  </div>\r\n</div>\r\n<div class=\"modal-footer justify-content-start\">\r\n  <button class=\"btn btn-primary\" [disabled]=\"contactForm.invalid\" (click)=\"onFormSubmit()\">\r\n    <gx-icon name=\"contacts\" [fw]=\"true\"></gx-icon>\r\n    <span> Save Contact </span>\r\n  </button>\r\n</div>\r\n");

/***/ }),

/***/ "./src/app/content/apps/contacts/contact.model.ts":
/*!********************************************************!*\
  !*** ./src/app/content/apps/contacts/contact.model.ts ***!
  \********************************************************/
/*! exports provided: Contact */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Contact", function() { return Contact; });
/* harmony import */ var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! tslib */ "./node_modules/tslib/tslib.es6.js");

class Contact {
}


/***/ }),

/***/ "./src/app/content/apps/contacts/contacts-header/contacts-header.component.scss":
/*!**************************************************************************************!*\
  !*** ./src/app/content/apps/contacts/contacts-header/contacts-header.component.scss ***!
  \**************************************************************************************/
/*! exports provided: default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony default export */ __webpack_exports__["default"] = ("\n/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IiIsImZpbGUiOiJzcmMvYXBwL2NvbnRlbnQvYXBwcy9jb250YWN0cy9jb250YWN0cy1oZWFkZXIvY29udGFjdHMtaGVhZGVyLmNvbXBvbmVudC5zY3NzIn0= */");

/***/ }),

/***/ "./src/app/content/apps/contacts/contacts-header/contacts-header.component.ts":
/*!************************************************************************************!*\
  !*** ./src/app/content/apps/contacts/contacts-header/contacts-header.component.ts ***!
  \************************************************************************************/
/*! exports provided: ContactsHeaderComponent */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "ContactsHeaderComponent", function() { return ContactsHeaderComponent; });
/* harmony import */ var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! tslib */ "./node_modules/tslib/tslib.es6.js");
/* harmony import */ var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @angular/core */ "./node_modules/@angular/core/fesm2015/core.js");
/* harmony import */ var _app_app_service__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @app/app.service */ "./src/app/app.service.ts");



let ContactsHeaderComponent = class ContactsHeaderComponent {
    constructor(appService) {
        this.appService = appService;
        this.classlist = 'dt-module__header d-none d-md-flex';
        this.notifications = [];
        this.onNotificationsChanged = this.appService.onNotificationsChanged.subscribe((notifications) => {
            this.notifications = notifications;
        });
    }
    ngOnInit() {
    }
    ngOnDestroy() {
        this.onNotificationsChanged.unsubscribe();
    }
};
ContactsHeaderComponent.ctorParameters = () => [
    { type: _app_app_service__WEBPACK_IMPORTED_MODULE_2__["AppService"] }
];
tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
    Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["HostBinding"])('class')
], ContactsHeaderComponent.prototype, "classlist", void 0);
ContactsHeaderComponent = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
    Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Component"])({
        selector: 'app-contacts-header',
        template: tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(/*! raw-loader!./contacts-header.component.html */ "./node_modules/raw-loader/dist/cjs.js!./src/app/content/apps/contacts/contacts-header/contacts-header.component.html")).default,
        styles: [tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(/*! ./contacts-header.component.scss */ "./src/app/content/apps/contacts/contacts-header/contacts-header.component.scss")).default]
    })
], ContactsHeaderComponent);



/***/ }),

/***/ "./src/app/content/apps/contacts/contacts-header/search-bar/search-bar.component.ts":
/*!******************************************************************************************!*\
  !*** ./src/app/content/apps/contacts/contacts-header/search-bar/search-bar.component.ts ***!
  \******************************************************************************************/
/*! exports provided: SearchBarComponent */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "SearchBarComponent", function() { return SearchBarComponent; });
/* harmony import */ var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! tslib */ "./node_modules/tslib/tslib.es6.js");
/* harmony import */ var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @angular/core */ "./node_modules/@angular/core/fesm2015/core.js");
/* harmony import */ var _angular_forms__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @angular/forms */ "./node_modules/@angular/forms/fesm2015/forms.js");
/* harmony import */ var _contacts_service__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ./../../contacts.service */ "./src/app/content/apps/contacts/contacts.service.ts");
/* harmony import */ var rxjs_operators__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! rxjs/operators */ "./node_modules/rxjs/_esm2015/operators/index.js");





let SearchBarComponent = class SearchBarComponent {
    constructor(contactService) {
        this.contactService = contactService;
        this.classlist = 'search-box ml-n4';
        this.searchQuery = new _angular_forms__WEBPACK_IMPORTED_MODULE_2__["FormControl"]('');
    }
    ngOnInit() {
        this.searchQuery.valueChanges.pipe(Object(rxjs_operators__WEBPACK_IMPORTED_MODULE_4__["debounceTime"])(300), Object(rxjs_operators__WEBPACK_IMPORTED_MODULE_4__["distinctUntilChanged"])()).subscribe(searchText => {
            this.contactService.onSearchQueryChanged.next(searchText);
        });
    }
};
SearchBarComponent.ctorParameters = () => [
    { type: _contacts_service__WEBPACK_IMPORTED_MODULE_3__["ContactsService"] }
];
tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
    Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["HostBinding"])('class')
], SearchBarComponent.prototype, "classlist", void 0);
SearchBarComponent = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
    Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Component"])({
        selector: 'contact-search-bar',
        template: tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(/*! raw-loader!./search-bar.component.html */ "./node_modules/raw-loader/dist/cjs.js!./src/app/content/apps/contacts/contacts-header/search-bar/search-bar.component.html")).default
    })
], SearchBarComponent);



/***/ }),

/***/ "./src/app/content/apps/contacts/contacts-list/contact-list-header/contact-list-header.component.scss":
/*!************************************************************************************************************!*\
  !*** ./src/app/content/apps/contacts/contacts-list/contact-list-header/contact-list-header.component.scss ***!
  \************************************************************************************************************/
/*! exports provided: default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony default export */ __webpack_exports__["default"] = (":host {\n  display: block; }\n\n/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbInNyYy9hcHAvY29udGVudC9hcHBzL2NvbnRhY3RzL2NvbnRhY3RzLWxpc3QvY29udGFjdC1saXN0LWhlYWRlci9DOlxcVXNlcnNcXHZpbm9cXHZpbm90aFxcdHJ1c3RwcmltZS12My4xXFx0cnVzdC1wcmltZS12My9zcmNcXGFwcFxcY29udGVudFxcYXBwc1xcY29udGFjdHNcXGNvbnRhY3RzLWxpc3RcXGNvbnRhY3QtbGlzdC1oZWFkZXJcXGNvbnRhY3QtbGlzdC1oZWFkZXIuY29tcG9uZW50LnNjc3MiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7RUFDRSxjQUFjLEVBQUEiLCJmaWxlIjoic3JjL2FwcC9jb250ZW50L2FwcHMvY29udGFjdHMvY29udGFjdHMtbGlzdC9jb250YWN0LWxpc3QtaGVhZGVyL2NvbnRhY3QtbGlzdC1oZWFkZXIuY29tcG9uZW50LnNjc3MiLCJzb3VyY2VzQ29udGVudCI6WyI6aG9zdHtcclxuICBkaXNwbGF5OiBibG9jaztcclxufVxyXG4iXX0= */");

/***/ }),

/***/ "./src/app/content/apps/contacts/contacts-list/contact-list-header/contact-list-header.component.ts":
/*!**********************************************************************************************************!*\
  !*** ./src/app/content/apps/contacts/contacts-list/contact-list-header/contact-list-header.component.ts ***!
  \**********************************************************************************************************/
/*! exports provided: ContactListHeaderComponent */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "ContactListHeaderComponent", function() { return ContactListHeaderComponent; });
/* harmony import */ var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! tslib */ "./node_modules/tslib/tslib.es6.js");
/* harmony import */ var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @angular/core */ "./node_modules/@angular/core/fesm2015/core.js");
/* harmony import */ var _contacts_service__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ./../../contacts.service */ "./src/app/content/apps/contacts/contacts.service.ts");



let ContactListHeaderComponent = class ContactListHeaderComponent {
    constructor(contactService) {
        this.contactService = contactService;
        this.classlist = 'px-1 pb-4 border-bottom border-w-2 mb-1 mt-n1';
        this.selected = 'None';
    }
    ngOnInit() {
        // Subscribe to update SelectedMails on changes
        this.onSelectedContactsChanged =
            this.contactService.onSelectedContactsChanged
                .subscribe(selectedContacts => {
                setTimeout(() => {
                    this.hasSelectedContacts = selectedContacts.length > 0;
                    this.isIndeterminate = (selectedContacts.length !== this.contactService.contacts.length && selectedContacts.length > 0);
                    if (selectedContacts.length === 0) {
                        this.selected = 'None';
                    }
                    else if (selectedContacts.length === this.contactService.contacts.length) {
                        this.selected = 'All';
                    }
                }, 0);
            });
    }
    /**
     * Toggle select all contacts
     */
    toggleSelectAll() {
        this.contactService.toggleSelectAll();
    }
    /**
     * Select contacts by filter
     * @param selected
     * @param filterParameter
     * @param filterValue
     */
    selectContacts(selected, filterParameter, filterValue) {
        this.selected = (selected) ? selected : this.selected;
        this.contactService.selectContacts(filterParameter, filterValue);
    }
    /**
     * Deselect all contacts
     */
    deselectContacts() {
        this.contactService.deselectContacts();
    }
    /**
     * Change folder to selected contacts
     * @param property
     * @param value
     */
    setPropertyOnSelectedContacts(property, value) {
        this.contactService.setPropertyOnSelectedContacts(property, value);
    }
    ngOnDestroy() {
        this.onSelectedContactsChanged.unsubscribe();
    }
};
ContactListHeaderComponent.ctorParameters = () => [
    { type: _contacts_service__WEBPACK_IMPORTED_MODULE_2__["ContactsService"] }
];
tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
    Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["HostBinding"])('class')
], ContactListHeaderComponent.prototype, "classlist", void 0);
ContactListHeaderComponent = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
    Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Component"])({
        selector: 'app-contact-list-header',
        template: tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(/*! raw-loader!./contact-list-header.component.html */ "./node_modules/raw-loader/dist/cjs.js!./src/app/content/apps/contacts/contacts-list/contact-list-header/contact-list-header.component.html")).default,
        styles: [tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(/*! ./contact-list-header.component.scss */ "./src/app/content/apps/contacts/contacts-list/contact-list-header/contact-list-header.component.scss")).default]
    })
], ContactListHeaderComponent);



/***/ }),

/***/ "./src/app/content/apps/contacts/contacts-list/contact-list-item/contact-list-item.component.scss":
/*!********************************************************************************************************!*\
  !*** ./src/app/content/apps/contacts/contacts-list/contact-list-item/contact-list-item.component.scss ***!
  \********************************************************************************************************/
/*! exports provided: default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony default export */ __webpack_exports__["default"] = ("\n/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IiIsImZpbGUiOiJzcmMvYXBwL2NvbnRlbnQvYXBwcy9jb250YWN0cy9jb250YWN0cy1saXN0L2NvbnRhY3QtbGlzdC1pdGVtL2NvbnRhY3QtbGlzdC1pdGVtLmNvbXBvbmVudC5zY3NzIn0= */");

/***/ }),

/***/ "./src/app/content/apps/contacts/contacts-list/contact-list-item/contact-list-item.component.ts":
/*!******************************************************************************************************!*\
  !*** ./src/app/content/apps/contacts/contacts-list/contact-list-item/contact-list-item.component.ts ***!
  \******************************************************************************************************/
/*! exports provided: ContactListItemComponent */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "ContactListItemComponent", function() { return ContactListItemComponent; });
/* harmony import */ var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! tslib */ "./node_modules/tslib/tslib.es6.js");
/* harmony import */ var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @angular/core */ "./node_modules/@angular/core/fesm2015/core.js");
/* harmony import */ var _ng_bootstrap_ng_bootstrap__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @ng-bootstrap/ng-bootstrap */ "./node_modules/@ng-bootstrap/ng-bootstrap/fesm2015/ng-bootstrap.js");
/* harmony import */ var _contacts_service__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ./../../contacts.service */ "./src/app/content/apps/contacts/contacts.service.ts");
/* harmony import */ var _edit_contact_edit_contact_component__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ./../../edit-contact/edit-contact.component */ "./src/app/content/apps/contacts/edit-contact/edit-contact.component.ts");





'';
let ContactListItemComponent = class ContactListItemComponent {
    constructor(contactService, modalService) {
        this.contactService = contactService;
        this.modalService = modalService;
        this.classlist = 'dt-module__list-item';
    }
    ngOnInit() {
        // Subscribe to update SelectedMails on changes
        this.onSelectedContactsChanged =
            this.contactService.onSelectedContactsChanged
                .subscribe(selectedContacts => {
                this.selected = selectedContacts.find((contact) => {
                    return contact.id === this.contact.id;
                });
            });
    }
    onSelectedChange(event) {
        this.selected = event;
        this.contactService.toggleSelectedContact(this.contact.id);
    }
    /**
     * On toggle starred
     */
    onToggleStarred(event) {
        this.contact.starred = event;
        this.contactService.saveContact(this.contact);
    }
    /**
     * Edit a contact
     * @param {Contact} contact
     */
    editContact(contact) {
        const modalRef = this.modalService.open(_edit_contact_edit_contact_component__WEBPACK_IMPORTED_MODULE_4__["EditContactComponent"], { size: 'lg' });
        modalRef.componentInstance.contact = contact;
        modalRef.componentInstance.title = 'Edit Contact';
    }
    /**
     * Delete a contact
     * @param {Contact} contact
     */
    deleteContact(contact) {
        this.contactService.deleteContact(contact.id);
    }
    /**
     * Add contact in Favourite List
     */
    onFavouriteContact() {
        this.contact.favourite = true;
        this.contactService.saveContact(this.contact);
    }
    ngOnDestroy() {
        this.onSelectedContactsChanged.unsubscribe();
    }
};
ContactListItemComponent.ctorParameters = () => [
    { type: _contacts_service__WEBPACK_IMPORTED_MODULE_3__["ContactsService"] },
    { type: _ng_bootstrap_ng_bootstrap__WEBPACK_IMPORTED_MODULE_2__["NgbModal"] }
];
tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
    Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["HostBinding"])('class')
], ContactListItemComponent.prototype, "classlist", void 0);
tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
    Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["HostBinding"])('class.selected')
], ContactListItemComponent.prototype, "selected", void 0);
tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
    Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Input"])()
], ContactListItemComponent.prototype, "contact", void 0);
ContactListItemComponent = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
    Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Component"])({
        selector: 'app-contact-list-item',
        template: tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(/*! raw-loader!./contact-list-item.component.html */ "./node_modules/raw-loader/dist/cjs.js!./src/app/content/apps/contacts/contacts-list/contact-list-item/contact-list-item.component.html")).default,
        styles: [tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(/*! ./contact-list-item.component.scss */ "./src/app/content/apps/contacts/contacts-list/contact-list-item/contact-list-item.component.scss")).default]
    })
], ContactListItemComponent);



/***/ }),

/***/ "./src/app/content/apps/contacts/contacts-list/contacts-list.component.scss":
/*!**********************************************************************************!*\
  !*** ./src/app/content/apps/contacts/contacts-list/contacts-list.component.scss ***!
  \**********************************************************************************/
/*! exports provided: default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony default export */ __webpack_exports__["default"] = ("\n/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IiIsImZpbGUiOiJzcmMvYXBwL2NvbnRlbnQvYXBwcy9jb250YWN0cy9jb250YWN0cy1saXN0L2NvbnRhY3RzLWxpc3QuY29tcG9uZW50LnNjc3MifQ== */");

/***/ }),

/***/ "./src/app/content/apps/contacts/contacts-list/contacts-list.component.ts":
/*!********************************************************************************!*\
  !*** ./src/app/content/apps/contacts/contacts-list/contacts-list.component.ts ***!
  \********************************************************************************/
/*! exports provided: ContactsListComponent */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "ContactsListComponent", function() { return ContactsListComponent; });
/* harmony import */ var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! tslib */ "./node_modules/tslib/tslib.es6.js");
/* harmony import */ var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @angular/core */ "./node_modules/@angular/core/fesm2015/core.js");
/* harmony import */ var _contacts_service__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ./../contacts.service */ "./src/app/content/apps/contacts/contacts.service.ts");
/* harmony import */ var ngx_perfect_scrollbar__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ngx-perfect-scrollbar */ "./node_modules/ngx-perfect-scrollbar/dist/ngx-perfect-scrollbar.es5.js");




let ContactsListComponent = class ContactsListComponent {
    constructor(contactService) {
        this.contactService = contactService;
        this.classlist = 'dt-module__content position-relative ps';
        this.contacts = [];
        this.searchQuery = '';
    }
    ngOnInit() {
        // Subscribe to update mails on changes
        this.onContactsChanged = this.contactService.onContactsChanged
            .subscribe(contacts => {
            this.contacts = contacts;
            setTimeout(() => {
                this.scrollToBottom();
            }, 300);
        });
        // Subscribe to update search keywords on changes
        this.onSearchQueryChanged = this.contactService.onSearchQueryChanged
            .subscribe(searchStr => {
            this.searchQuery = searchStr;
        });
    }
    scrollToBottom() {
        if (this.directiveRef) {
            this.directiveRef.update();
            this.directiveRef.scrollToBottom();
        }
    }
    ngOnDestroy() {
        this.onContactsChanged.unsubscribe();
        this.onSearchQueryChanged.unsubscribe();
    }
};
ContactsListComponent.ctorParameters = () => [
    { type: _contacts_service__WEBPACK_IMPORTED_MODULE_2__["ContactsService"] }
];
tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
    Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["HostBinding"])('class')
], ContactsListComponent.prototype, "classlist", void 0);
tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
    Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["ViewChild"])(ngx_perfect_scrollbar__WEBPACK_IMPORTED_MODULE_3__["PerfectScrollbarDirective"], { static: false })
], ContactsListComponent.prototype, "directiveRef", void 0);
ContactsListComponent = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
    Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Component"])({
        selector: 'app-contacts-list',
        template: tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(/*! raw-loader!./contacts-list.component.html */ "./node_modules/raw-loader/dist/cjs.js!./src/app/content/apps/contacts/contacts-list/contacts-list.component.html")).default,
        styles: [tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(/*! ./contacts-list.component.scss */ "./src/app/content/apps/contacts/contacts-list/contacts-list.component.scss")).default]
    })
], ContactsListComponent);



/***/ }),

/***/ "./src/app/content/apps/contacts/contacts-sidebar/contacts-sidebar.component.ts":
/*!**************************************************************************************!*\
  !*** ./src/app/content/apps/contacts/contacts-sidebar/contacts-sidebar.component.ts ***!
  \**************************************************************************************/
/*! exports provided: ContactsSidebarComponent */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "ContactsSidebarComponent", function() { return ContactsSidebarComponent; });
/* harmony import */ var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! tslib */ "./node_modules/tslib/tslib.es6.js");
/* harmony import */ var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @angular/core */ "./node_modules/@angular/core/fesm2015/core.js");
/* harmony import */ var _angular_router__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @angular/router */ "./node_modules/@angular/router/fesm2015/router.js");
/* harmony import */ var _ng_bootstrap_ng_bootstrap__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! @ng-bootstrap/ng-bootstrap */ "./node_modules/@ng-bootstrap/ng-bootstrap/fesm2015/ng-bootstrap.js");
/* harmony import */ var _gaxon_mix_animations__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! @gaxon/mix/animations */ "./src/gaxon/mix/animations.ts");
/* harmony import */ var _drawer_service__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ./../../drawer.service */ "./src/app/content/apps/drawer.service.ts");
/* harmony import */ var _app_settings_settings_service__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! @app/settings/settings.service */ "./src/app/settings/settings.service.ts");
/* harmony import */ var _gaxon_helpers__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! @gaxon/helpers */ "./src/gaxon/helpers/index.ts");
/* harmony import */ var _edit_contact_edit_contact_component__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! ./../edit-contact/edit-contact.component */ "./src/app/content/apps/contacts/edit-contact/edit-contact.component.ts");
/* harmony import */ var _contact_model__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! ./../contact.model */ "./src/app/content/apps/contacts/contact.model.ts");










let ContactsSidebarComponent = class ContactsSidebarComponent {
    constructor(modalService, drawerService, settingService, router) {
        this.modalService = modalService;
        this.drawerService = drawerService;
        this.settingService = settingService;
        this.router = router;
        this.classNames = 'dt-module__sidebar';
        this.isOpen = false;
        this.headerLessLayout = ['modern', 'back-office', 'back-office-mini'];
        this.onDrawerChanged = this.drawerService.onDrawerChanged.subscribe((drawer) => {
            this.drawer = drawer;
            this.activeMenu = this.drawer.activeMenu;
            if (!this.drawer.isOpen) {
                this.activeMenu = '';
            }
        });
        this.onSettingChanged = this.settingService.onSettingChanged.subscribe((newSettings) => {
            this.settings = newSettings;
            this.hiddenPageHeader = this.headerLessLayout.includes(this.settings.layout);
        });
        router.events.subscribe((event) => {
            if (event instanceof _angular_router__WEBPACK_IMPORTED_MODULE_2__["NavigationEnd"]) {
                this.isOpen = window.innerWidth > 767;
            }
        });
    }
    onResize(event) {
        this.isOpen = event.target.innerWidth > 767;
    }
    ngOnInit() {
        // display app sidebar
        this.isOpen = window.innerWidth > 767;
    }
    onClickNewContact() {
        const modalRef = this.modalService.open(_edit_contact_edit_contact_component__WEBPACK_IMPORTED_MODULE_8__["EditContactComponent"], { size: 'lg', windowClass: 'contact-modal' });
        modalRef.componentInstance.contact = this.createNewContact();
    }
    createNewContact() {
        const newContact = new _contact_model__WEBPACK_IMPORTED_MODULE_9__["Contact"]();
        newContact.id = _gaxon_helpers__WEBPACK_IMPORTED_MODULE_7__["GxHelper"].generateGUID();
        return newContact;
    }
    /**
     * On Click Menu
     */
    onClickMenu(event, activeMenu) {
        event.preventDefault();
        this.drawer.activeMenu = this.activeMenu = activeMenu;
        this.drawer.isOpen = true;
        this.drawerService.update(this.drawer);
    }
    /**
     * Toggle collapse
     *
     * @param ev
     */
    toggleOpen(ev) {
        ev.preventDefault();
        this.isOpen = !this.isOpen;
    }
    /**
     * Toggle main navigation
     * @param event
     */
    toggleNavigation(event) {
        event.preventDefault();
        this.settings.activeNavDrawer = !this.settings.activeNavDrawer;
        this.settingService.setSettings(this.settings);
    }
    ngOnDestroy() {
        this.onDrawerChanged.unsubscribe();
        this.onSettingChanged.unsubscribe();
    }
};
ContactsSidebarComponent.ctorParameters = () => [
    { type: _ng_bootstrap_ng_bootstrap__WEBPACK_IMPORTED_MODULE_3__["NgbModal"] },
    { type: _drawer_service__WEBPACK_IMPORTED_MODULE_5__["DrawerService"] },
    { type: _app_settings_settings_service__WEBPACK_IMPORTED_MODULE_6__["SettingsService"] },
    { type: _angular_router__WEBPACK_IMPORTED_MODULE_2__["Router"] }
];
tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
    Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["HostBinding"])('class')
], ContactsSidebarComponent.prototype, "classNames", void 0);
tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
    Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["HostBinding"])('class.active')
], ContactsSidebarComponent.prototype, "isOpen", void 0);
tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
    Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["HostListener"])('window:resize', ['$event'])
], ContactsSidebarComponent.prototype, "onResize", null);
ContactsSidebarComponent = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
    Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Component"])({
        selector: 'app-contacts-sidebar',
        template: tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(/*! raw-loader!./contacts-sidebar.component.html */ "./node_modules/raw-loader/dist/cjs.js!./src/app/content/apps/contacts/contacts-sidebar/contacts-sidebar.component.html")).default,
        animations: _gaxon_mix_animations__WEBPACK_IMPORTED_MODULE_4__["appAnimations"]
    })
], ContactsSidebarComponent);



/***/ }),

/***/ "./src/app/content/apps/contacts/contacts.component.scss":
/*!***************************************************************!*\
  !*** ./src/app/content/apps/contacts/contacts.component.scss ***!
  \***************************************************************/
/*! exports provided: default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony default export */ __webpack_exports__["default"] = ("\n/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IiIsImZpbGUiOiJzcmMvYXBwL2NvbnRlbnQvYXBwcy9jb250YWN0cy9jb250YWN0cy5jb21wb25lbnQuc2NzcyJ9 */");

/***/ }),

/***/ "./src/app/content/apps/contacts/contacts.component.ts":
/*!*************************************************************!*\
  !*** ./src/app/content/apps/contacts/contacts.component.ts ***!
  \*************************************************************/
/*! exports provided: ContactsComponent */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "ContactsComponent", function() { return ContactsComponent; });
/* harmony import */ var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! tslib */ "./node_modules/tslib/tslib.es6.js");
/* harmony import */ var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @angular/core */ "./node_modules/@angular/core/fesm2015/core.js");
/* harmony import */ var _drawer_service__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ./../drawer.service */ "./src/app/content/apps/drawer.service.ts");



let ContactsComponent = class ContactsComponent {
    constructor(drawerService) {
        this.drawerService = drawerService;
        this.onDrawerChanged = this.drawerService.onDrawerChanged.subscribe((drawer) => {
            this.drawer = drawer;
        });
    }
    ngOnInit() {
    }
    /**
     * Close Drawer
     * @param event
     */
    closeDrawer(event) {
        event.preventDefault();
        this.drawer.isOpen = false;
        this.drawerService.update(this.drawer);
    }
    ngOnDestroy() {
        this.onDrawerChanged.unsubscribe();
    }
};
ContactsComponent.ctorParameters = () => [
    { type: _drawer_service__WEBPACK_IMPORTED_MODULE_2__["DrawerService"] }
];
ContactsComponent = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
    Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Component"])({
        selector: 'app-contacts',
        template: tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(/*! raw-loader!./contacts.component.html */ "./node_modules/raw-loader/dist/cjs.js!./src/app/content/apps/contacts/contacts.component.html")).default,
        styles: [tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(/*! ./contacts.component.scss */ "./src/app/content/apps/contacts/contacts.component.scss")).default]
    })
], ContactsComponent);



/***/ }),

/***/ "./src/app/content/apps/contacts/contacts.module.ts":
/*!**********************************************************!*\
  !*** ./src/app/content/apps/contacts/contacts.module.ts ***!
  \**********************************************************/
/*! exports provided: ContactsModule */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "ContactsModule", function() { return ContactsModule; });
/* harmony import */ var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! tslib */ "./node_modules/tslib/tslib.es6.js");
/* harmony import */ var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @angular/core */ "./node_modules/@angular/core/fesm2015/core.js");
/* harmony import */ var _angular_common__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @angular/common */ "./node_modules/@angular/common/fesm2015/common.js");
/* harmony import */ var _gaxon_modules__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! @gaxon/modules */ "./src/gaxon/modules/index.ts");
/* harmony import */ var _angular_router__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! @angular/router */ "./node_modules/@angular/router/fesm2015/router.js");
/* harmony import */ var ngx_perfect_scrollbar__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ngx-perfect-scrollbar */ "./node_modules/ngx-perfect-scrollbar/dist/ngx-perfect-scrollbar.es5.js");
/* harmony import */ var _contacts_service__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ./contacts.service */ "./src/app/content/apps/contacts/contacts.service.ts");
/* harmony import */ var _contacts_component__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ./contacts.component */ "./src/app/content/apps/contacts/contacts.component.ts");
/* harmony import */ var _contacts_list_contacts_list_component__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! ./contacts-list/contacts-list.component */ "./src/app/content/apps/contacts/contacts-list/contacts-list.component.ts");
/* harmony import */ var _contacts_list_contact_list_item_contact_list_item_component__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! ./contacts-list/contact-list-item/contact-list-item.component */ "./src/app/content/apps/contacts/contacts-list/contact-list-item/contact-list-item.component.ts");
/* harmony import */ var _contacts_sidebar_contacts_sidebar_component__WEBPACK_IMPORTED_MODULE_10__ = __webpack_require__(/*! ./contacts-sidebar/contacts-sidebar.component */ "./src/app/content/apps/contacts/contacts-sidebar/contacts-sidebar.component.ts");
/* harmony import */ var _contacts_header_contacts_header_component__WEBPACK_IMPORTED_MODULE_11__ = __webpack_require__(/*! ./contacts-header/contacts-header.component */ "./src/app/content/apps/contacts/contacts-header/contacts-header.component.ts");
/* harmony import */ var _contacts_header_search_bar_search_bar_component__WEBPACK_IMPORTED_MODULE_12__ = __webpack_require__(/*! ./contacts-header/search-bar/search-bar.component */ "./src/app/content/apps/contacts/contacts-header/search-bar/search-bar.component.ts");
/* harmony import */ var _contacts_list_contact_list_header_contact_list_header_component__WEBPACK_IMPORTED_MODULE_13__ = __webpack_require__(/*! ./contacts-list/contact-list-header/contact-list-header.component */ "./src/app/content/apps/contacts/contacts-list/contact-list-header/contact-list-header.component.ts");
/* harmony import */ var _edit_contact_edit_contact_component__WEBPACK_IMPORTED_MODULE_14__ = __webpack_require__(/*! ./edit-contact/edit-contact.component */ "./src/app/content/apps/contacts/edit-contact/edit-contact.component.ts");
/* harmony import */ var _drawer_drawer_component__WEBPACK_IMPORTED_MODULE_15__ = __webpack_require__(/*! ./drawer/drawer.component */ "./src/app/content/apps/contacts/drawer/drawer.component.ts");
/* harmony import */ var _drawer_notifications_notifications_component__WEBPACK_IMPORTED_MODULE_16__ = __webpack_require__(/*! ./drawer/notifications/notifications.component */ "./src/app/content/apps/contacts/drawer/notifications/notifications.component.ts");
/* harmony import */ var _drawer_search_search_component__WEBPACK_IMPORTED_MODULE_17__ = __webpack_require__(/*! ./drawer/search/search.component */ "./src/app/content/apps/contacts/drawer/search/search.component.ts");


















const DEFAULT_PERFECT_SCROLLBAR_CONFIG = {
    suppressScrollX: true,
    wheelPropagation: true
};
const routes = [
    {
        path: 'all',
        component: _contacts_component__WEBPACK_IMPORTED_MODULE_7__["ContactsComponent"],
        resolve: {
            contacts: _contacts_service__WEBPACK_IMPORTED_MODULE_6__["ContactsService"]
        }
    },
    {
        path: ':handle',
        component: _contacts_component__WEBPACK_IMPORTED_MODULE_7__["ContactsComponent"],
        resolve: {
            contacts: _contacts_service__WEBPACK_IMPORTED_MODULE_6__["ContactsService"]
        }
    },
    {
        path: '**',
        redirectTo: 'all'
    }
];
let ContactsModule = class ContactsModule {
};
ContactsModule = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
    Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["NgModule"])({
        imports: [
            _angular_router__WEBPACK_IMPORTED_MODULE_4__["RouterModule"].forChild(routes),
            _angular_common__WEBPACK_IMPORTED_MODULE_2__["CommonModule"],
            _gaxon_modules__WEBPACK_IMPORTED_MODULE_3__["SharedModule"],
            ngx_perfect_scrollbar__WEBPACK_IMPORTED_MODULE_5__["PerfectScrollbarModule"]
        ],
        declarations: [
            _contacts_component__WEBPACK_IMPORTED_MODULE_7__["ContactsComponent"],
            _contacts_list_contacts_list_component__WEBPACK_IMPORTED_MODULE_8__["ContactsListComponent"],
            _contacts_list_contact_list_item_contact_list_item_component__WEBPACK_IMPORTED_MODULE_9__["ContactListItemComponent"],
            _contacts_sidebar_contacts_sidebar_component__WEBPACK_IMPORTED_MODULE_10__["ContactsSidebarComponent"],
            _contacts_header_contacts_header_component__WEBPACK_IMPORTED_MODULE_11__["ContactsHeaderComponent"],
            _contacts_header_search_bar_search_bar_component__WEBPACK_IMPORTED_MODULE_12__["SearchBarComponent"],
            _contacts_list_contact_list_header_contact_list_header_component__WEBPACK_IMPORTED_MODULE_13__["ContactListHeaderComponent"],
            _edit_contact_edit_contact_component__WEBPACK_IMPORTED_MODULE_14__["EditContactComponent"],
            _drawer_drawer_component__WEBPACK_IMPORTED_MODULE_15__["DrawerComponent"],
            _drawer_notifications_notifications_component__WEBPACK_IMPORTED_MODULE_16__["NotificationsComponent"],
            _drawer_search_search_component__WEBPACK_IMPORTED_MODULE_17__["SearchComponent"]
        ],
        entryComponents: [
            _edit_contact_edit_contact_component__WEBPACK_IMPORTED_MODULE_14__["EditContactComponent"]
        ],
        providers: [
            {
                provide: ngx_perfect_scrollbar__WEBPACK_IMPORTED_MODULE_5__["PERFECT_SCROLLBAR_CONFIG"],
                useValue: DEFAULT_PERFECT_SCROLLBAR_CONFIG
            }
        ]
    })
], ContactsModule);



/***/ }),

/***/ "./src/app/content/apps/contacts/contacts.service.ts":
/*!***********************************************************!*\
  !*** ./src/app/content/apps/contacts/contacts.service.ts ***!
  \***********************************************************/
/*! exports provided: ContactsService */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "ContactsService", function() { return ContactsService; });
/* harmony import */ var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! tslib */ "./node_modules/tslib/tslib.es6.js");
/* harmony import */ var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @angular/core */ "./node_modules/@angular/core/fesm2015/core.js");
/* harmony import */ var _angular_common__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @angular/common */ "./node_modules/@angular/common/fesm2015/common.js");
/* harmony import */ var _angular_common_http__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! @angular/common/http */ "./node_modules/@angular/common/fesm2015/http.js");
/* harmony import */ var rxjs__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! rxjs */ "./node_modules/rxjs/_esm2015/index.js");





let ContactsService = class ContactsService {
    constructor(http, location) {
        this.http = http;
        this.location = location;
        this.contactsUrl = 'api/contacts'; // URL to web api
        this.selectedContacts = [];
        this.onContactsChanged = new rxjs__WEBPACK_IMPORTED_MODULE_4__["BehaviorSubject"]([]);
        this.onSelectedContactsChanged = new rxjs__WEBPACK_IMPORTED_MODULE_4__["BehaviorSubject"]([]);
        this.onSearchQueryChanged = new rxjs__WEBPACK_IMPORTED_MODULE_4__["BehaviorSubject"]('');
    }
    /**
     * Resolve
     * @param {ActivatedRouteSnapshot} route
     * @param {RouterStateSnapshot} state
     * @returns {Observable<any> | Promise<any> | any}
     */
    resolve(route, state) {
        this.routeParams = route.params;
        return new Promise((resolve, reject) => {
            Promise.all([
                this.getContacts(),
            ]).then(() => {
                resolve();
            }, reject);
        });
    }
    /**
     * Get contacts
     * @returns {Promise<Contact[]>}
     */
    getContacts() {
        if (this.routeParams.handle) {
            return this.getContactsByParams(this.routeParams.handle);
        }
        else {
            return this.getAllContacts();
        }
    }
    /**
     * Get all contacts
     * @returns {Promise<Contact[]>}
     */
    getAllContacts() {
        return new Promise((resolve, reject) => {
            this.http.get(this.contactsUrl)
                .subscribe((response) => {
                this.contacts = response;
                this.onContactsChanged.next(this.contacts);
                resolve(this.contacts);
            }, reject);
        });
    }
    /**
     * Get contacts by param
     * @param param
     * @return {Promise<Contact[]>}
     */
    getContactsByParams(param) {
        const params = '?' + param + '=true';
        return new Promise((resolve, reject) => {
            this.http.get(this.contactsUrl + params)
                .subscribe((response) => {
                this.contacts = response;
                this.onContactsChanged.next(this.contacts);
                resolve(this.contacts);
            }, reject);
        });
    }
    /**
     * Toggle selected contact by id
     * @param id
     */
    toggleSelectedContact(id) {
        // First, check if we already have that contact as selected
        if (this.selectedContacts.length > 0) {
            for (const contact of this.selectedContacts) {
                // delete the selected contact
                if (contact.id === id) {
                    const index = this.selectedContacts.indexOf(contact);
                    if (index !== -1) {
                        this.selectedContacts.splice(index, 1);
                        // Trigger the next event
                        this.onSelectedContactsChanged.next(this.selectedContacts);
                        // Return
                        return;
                    }
                }
            }
        }
        // If we don't have it, push in selected list
        this.selectedContacts.push(this.contacts.find(contact => {
            return contact.id === id;
        }));
        // Trigger the next event
        this.onSelectedContactsChanged.next(this.selectedContacts);
    }
    /**
     * Toggle select all
     */
    toggleSelectAll() {
        if (this.selectedContacts.length > 0) {
            this.deselectContacts();
        }
        else {
            this.selectContacts();
        }
    }
    /**
     * Add contact in selectedContacts list
     * @param filterParameter
     * @param filterValue
     */
    selectContacts(filterParameter, filterValue) {
        this.selectedContacts = [];
        // If there is no filter, select all mails
        if (filterParameter === undefined || filterValue === undefined) {
            this.selectedContacts = this.contacts.map(x => Object.assign({}, x));
        }
        else {
            this.selectedContacts.push(...this.contacts.filter(contact => {
                return contact[filterParameter] === filterValue;
            }));
        }
        // Trigger the next event
        this.onSelectedContactsChanged.next(this.selectedContacts);
    }
    /**
     * Deselect all contacts
     */
    deselectContacts() {
        this.selectedContacts = [];
        // Trigger the next event
        this.onSelectedContactsChanged.next(this.selectedContacts);
    }
    /**
     * Set property on selected contacts
     * @param property
     * @param value
     */
    setPropertyOnSelectedContacts(property, value) {
        this.selectedContacts.map(contact => {
            contact[property] = value;
            this.saveContact(contact);
        });
        this.deselectContacts();
    }
    /**
     * Save a contact
     * @param {Contact} contact
     * @return {Promise<any>}
     */
    saveContact(contact) {
        return new Promise((resolve, reject) => {
            this.http.post(this.contactsUrl + '/' + contact.id, Object.assign({}, contact))
                .subscribe(response => {
                this.getContacts().then(contacts => {
                    resolve(contacts);
                }, reject);
            });
        });
    }
    /**
     * Delete a contact from data
     * @param contactId
     * @return {Promise<any>}
     */
    deleteContact(contactId) {
        return new Promise((resolve, reject) => {
            this.http.delete(this.contactsUrl + '/' + contactId)
                .subscribe(response => {
                this.getContacts().then(contacts => {
                    resolve(contacts);
                }, reject);
            });
        });
    }
};
ContactsService.ctorParameters = () => [
    { type: _angular_common_http__WEBPACK_IMPORTED_MODULE_3__["HttpClient"] },
    { type: _angular_common__WEBPACK_IMPORTED_MODULE_2__["Location"] }
];
ContactsService = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
    Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Injectable"])({
        providedIn: 'root'
    })
], ContactsService);



/***/ }),

/***/ "./src/app/content/apps/contacts/drawer/drawer.component.scss":
/*!********************************************************************!*\
  !*** ./src/app/content/apps/contacts/drawer/drawer.component.scss ***!
  \********************************************************************/
/*! exports provided: default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony default export */ __webpack_exports__["default"] = ("\n/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IiIsImZpbGUiOiJzcmMvYXBwL2NvbnRlbnQvYXBwcy9jb250YWN0cy9kcmF3ZXIvZHJhd2VyLmNvbXBvbmVudC5zY3NzIn0= */");

/***/ }),

/***/ "./src/app/content/apps/contacts/drawer/drawer.component.ts":
/*!******************************************************************!*\
  !*** ./src/app/content/apps/contacts/drawer/drawer.component.ts ***!
  \******************************************************************/
/*! exports provided: DrawerComponent */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "DrawerComponent", function() { return DrawerComponent; });
/* harmony import */ var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! tslib */ "./node_modules/tslib/tslib.es6.js");
/* harmony import */ var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @angular/core */ "./node_modules/@angular/core/fesm2015/core.js");
/* harmony import */ var _drawer_service__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ./../../drawer.service */ "./src/app/content/apps/drawer.service.ts");



let DrawerComponent = class DrawerComponent {
    constructor(drawerService) {
        this.drawerService = drawerService;
        this.draverClasses = 'dt-module__drawer dt-drawer position-left';
        this.isOpen = false;
        this.showLoader = false;
        this.onDrawerChanged = this.drawerService.onDrawerChanged.subscribe((drawer) => {
            if (!this.activeMenu) {
                this.processLoader();
            }
            this.drawer = drawer;
            this.isOpen = this.isOpen;
            this.activeMenu = this.drawer.activeMenu;
            if (!this.drawer.isOpen) {
                this.activeMenu = '';
            }
        });
    }
    ngOnInit() {
    }
    processLoader() {
        this.showLoader = true;
        setTimeout(() => {
            this.showLoader = false;
        }, 1000);
    }
    /**
     * Close Drawer
     * @param event
     */
    closeDrawer(event) {
        event.preventDefault();
        this.drawer.isOpen = false;
        this.drawerService.update(this.drawer);
    }
    onSelectMenu(event, activeMenu) {
        event.preventDefault();
        this.processLoader();
        this.drawer.activeMenu = this.activeMenu = activeMenu;
        this.drawerService.update(this.drawer);
    }
    ngOnDestroy() {
        this.onDrawerChanged.unsubscribe();
    }
};
DrawerComponent.ctorParameters = () => [
    { type: _drawer_service__WEBPACK_IMPORTED_MODULE_2__["DrawerService"] }
];
tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
    Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["HostBinding"])('class')
], DrawerComponent.prototype, "draverClasses", void 0);
tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
    Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["HostBinding"])('class.open')
], DrawerComponent.prototype, "isOpen", void 0);
DrawerComponent = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
    Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Component"])({
        selector: 'contact-drawer',
        template: tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(/*! raw-loader!./drawer.component.html */ "./node_modules/raw-loader/dist/cjs.js!./src/app/content/apps/contacts/drawer/drawer.component.html")).default,
        styles: [tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(/*! ./drawer.component.scss */ "./src/app/content/apps/contacts/drawer/drawer.component.scss")).default]
    })
], DrawerComponent);



/***/ }),

/***/ "./src/app/content/apps/contacts/drawer/notifications/notifications.component.scss":
/*!*****************************************************************************************!*\
  !*** ./src/app/content/apps/contacts/drawer/notifications/notifications.component.scss ***!
  \*****************************************************************************************/
/*! exports provided: default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony default export */ __webpack_exports__["default"] = ("\n/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IiIsImZpbGUiOiJzcmMvYXBwL2NvbnRlbnQvYXBwcy9jb250YWN0cy9kcmF3ZXIvbm90aWZpY2F0aW9ucy9ub3RpZmljYXRpb25zLmNvbXBvbmVudC5zY3NzIn0= */");

/***/ }),

/***/ "./src/app/content/apps/contacts/drawer/notifications/notifications.component.ts":
/*!***************************************************************************************!*\
  !*** ./src/app/content/apps/contacts/drawer/notifications/notifications.component.ts ***!
  \***************************************************************************************/
/*! exports provided: NotificationsComponent */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "NotificationsComponent", function() { return NotificationsComponent; });
/* harmony import */ var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! tslib */ "./node_modules/tslib/tslib.es6.js");
/* harmony import */ var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @angular/core */ "./node_modules/@angular/core/fesm2015/core.js");
/* harmony import */ var _app_app_service__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @app/app.service */ "./src/app/app.service.ts");



let NotificationsComponent = class NotificationsComponent {
    constructor(appService) {
        this.appService = appService;
        this.classlist = 'dt-notification';
        this.notifications = [];
        this.onNotificationsChanged = this.appService.onNotificationsChanged.subscribe((notifications) => {
            this.notifications = notifications;
        });
    }
    ngOnInit() {
    }
    ngOnDestroy() {
        this.onNotificationsChanged.unsubscribe();
    }
};
NotificationsComponent.ctorParameters = () => [
    { type: _app_app_service__WEBPACK_IMPORTED_MODULE_2__["AppService"] }
];
tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
    Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["HostBinding"])('class')
], NotificationsComponent.prototype, "classlist", void 0);
NotificationsComponent = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
    Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Component"])({
        selector: 'todo-notifications',
        template: tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(/*! raw-loader!./notifications.component.html */ "./node_modules/raw-loader/dist/cjs.js!./src/app/content/apps/contacts/drawer/notifications/notifications.component.html")).default,
        styles: [tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(/*! ./notifications.component.scss */ "./src/app/content/apps/contacts/drawer/notifications/notifications.component.scss")).default]
    })
], NotificationsComponent);



/***/ }),

/***/ "./src/app/content/apps/contacts/drawer/search/search.component.scss":
/*!***************************************************************************!*\
  !*** ./src/app/content/apps/contacts/drawer/search/search.component.scss ***!
  \***************************************************************************/
/*! exports provided: default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony default export */ __webpack_exports__["default"] = ("\n/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IiIsImZpbGUiOiJzcmMvYXBwL2NvbnRlbnQvYXBwcy9jb250YWN0cy9kcmF3ZXIvc2VhcmNoL3NlYXJjaC5jb21wb25lbnQuc2NzcyJ9 */");

/***/ }),

/***/ "./src/app/content/apps/contacts/drawer/search/search.component.ts":
/*!*************************************************************************!*\
  !*** ./src/app/content/apps/contacts/drawer/search/search.component.ts ***!
  \*************************************************************************/
/*! exports provided: SearchComponent */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "SearchComponent", function() { return SearchComponent; });
/* harmony import */ var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! tslib */ "./node_modules/tslib/tslib.es6.js");
/* harmony import */ var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @angular/core */ "./node_modules/@angular/core/fesm2015/core.js");
/* harmony import */ var _angular_forms__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @angular/forms */ "./node_modules/@angular/forms/fesm2015/forms.js");
/* harmony import */ var _app_app_service__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! @app/app.service */ "./src/app/app.service.ts");
/* harmony import */ var rxjs_operators__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! rxjs/operators */ "./node_modules/rxjs/_esm2015/operators/index.js");





let SearchComponent = class SearchComponent {
    constructor(appService) {
        this.appService = appService;
        this.classlist = 'dt-notification';
        this.messages = [];
        this.searchText = '';
        this.searchQuery = new _angular_forms__WEBPACK_IMPORTED_MODULE_2__["FormControl"]('');
        this.onMessagesChanged = this.appService.onMessagesChanged.subscribe((messages) => {
            this.messages = messages;
        });
    }
    ngOnInit() {
        this.searchQuery.valueChanges.pipe(Object(rxjs_operators__WEBPACK_IMPORTED_MODULE_4__["debounceTime"])(300), Object(rxjs_operators__WEBPACK_IMPORTED_MODULE_4__["distinctUntilChanged"])()).subscribe(searchText => {
            this.searchText = searchText;
        });
    }
    ngOnDestroy() {
        this.onMessagesChanged.unsubscribe();
    }
};
SearchComponent.ctorParameters = () => [
    { type: _app_app_service__WEBPACK_IMPORTED_MODULE_3__["AppService"] }
];
tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
    Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["HostBinding"])('class')
], SearchComponent.prototype, "classlist", void 0);
SearchComponent = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
    Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Component"])({
        selector: 'todo-search',
        template: tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(/*! raw-loader!./search.component.html */ "./node_modules/raw-loader/dist/cjs.js!./src/app/content/apps/contacts/drawer/search/search.component.html")).default,
        styles: [tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(/*! ./search.component.scss */ "./src/app/content/apps/contacts/drawer/search/search.component.scss")).default]
    })
], SearchComponent);



/***/ }),

/***/ "./src/app/content/apps/contacts/edit-contact/edit-contact.component.scss":
/*!********************************************************************************!*\
  !*** ./src/app/content/apps/contacts/edit-contact/edit-contact.component.scss ***!
  \********************************************************************************/
/*! exports provided: default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony default export */ __webpack_exports__["default"] = ("\n/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IiIsImZpbGUiOiJzcmMvYXBwL2NvbnRlbnQvYXBwcy9jb250YWN0cy9lZGl0LWNvbnRhY3QvZWRpdC1jb250YWN0LmNvbXBvbmVudC5zY3NzIn0= */");

/***/ }),

/***/ "./src/app/content/apps/contacts/edit-contact/edit-contact.component.ts":
/*!******************************************************************************!*\
  !*** ./src/app/content/apps/contacts/edit-contact/edit-contact.component.ts ***!
  \******************************************************************************/
/*! exports provided: EditContactComponent */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "EditContactComponent", function() { return EditContactComponent; });
/* harmony import */ var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! tslib */ "./node_modules/tslib/tslib.es6.js");
/* harmony import */ var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @angular/core */ "./node_modules/@angular/core/fesm2015/core.js");
/* harmony import */ var _ng_bootstrap_ng_bootstrap__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @ng-bootstrap/ng-bootstrap */ "./node_modules/@ng-bootstrap/ng-bootstrap/fesm2015/ng-bootstrap.js");
/* harmony import */ var _contacts_service__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ./../contacts.service */ "./src/app/content/apps/contacts/contacts.service.ts");




let EditContactComponent = class EditContactComponent {
    constructor(activeModal, contactService) {
        this.activeModal = activeModal;
        this.contactService = contactService;
        this.title = 'Add Contact';
        this.imageSrc = '';
        this.filename = 'Change Thumb';
    }
    ngOnInit() {
    }
    handleInputChange(e) {
        const file = e.dataTransfer ? e.dataTransfer.files[0] : e.target.files[0];
        const pattern = /image-*/;
        const reader = new FileReader();
        if (!file.type.match(pattern)) {
            alert('invalid format');
            return;
        }
        this.filename = file.name;
        reader.onload = this._handleReaderLoaded.bind(this);
        reader.readAsDataURL(file);
    }
    /**
     * Handle file reader loaded
     * @param e
     * @private
     */
    _handleReaderLoaded(e) {
        const reader = e.target;
        this.contact.thumb = reader.result;
    }
    /**
     * On form submit call this function
     */
    onFormSubmit() {
        if (this.contactForm.valid) {
            this.contactService.saveContact(this.contact).then((response) => {
                this.contactForm.resetForm();
                this.activeModal.close();
            });
        }
    }
};
EditContactComponent.ctorParameters = () => [
    { type: _ng_bootstrap_ng_bootstrap__WEBPACK_IMPORTED_MODULE_2__["NgbActiveModal"] },
    { type: _contacts_service__WEBPACK_IMPORTED_MODULE_3__["ContactsService"] }
];
tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
    Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Input"])()
], EditContactComponent.prototype, "title", void 0);
tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
    Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Input"])()
], EditContactComponent.prototype, "contact", void 0);
tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
    Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["ViewChild"])('contactForm', { static: false })
], EditContactComponent.prototype, "contactForm", void 0);
EditContactComponent = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
    Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Component"])({
        selector: 'app-edit-contact',
        template: tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(/*! raw-loader!./edit-contact.component.html */ "./node_modules/raw-loader/dist/cjs.js!./src/app/content/apps/contacts/edit-contact/edit-contact.component.html")).default,
        styles: [tslib__WEBPACK_IMPORTED_MODULE_0__["__importDefault"](__webpack_require__(/*! ./edit-contact.component.scss */ "./src/app/content/apps/contacts/edit-contact/edit-contact.component.scss")).default]
    })
], EditContactComponent);



/***/ })

}]);
//# sourceMappingURL=contacts-contacts-module-es2015.js.map